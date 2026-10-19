#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integração com a API gratuita NASA POWER Climatology para expandir a cobertura
para qualquer um dos 5.570 municípios brasileiros via Latitude e Longitude.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import requests

from processador_clima_inmet import avaliar_mes, classificar_nota, MESES_ORDEM


def buscar_clima_municipio_nasa(lat: float, lon: float, alt_municipio: Optional[int] = None) -> Dict[str, Any]:
    """
    Consulta a API gratuita NASA POWER Climatology para obter médias mensais
    de Temperatura Média (T2M), Amplitude Térmica Diária Média (T2M_RANGE),
    Extremos Absolutos (T2M_MAX, T2M_MIN), Umidade Relativa (RH2M),
    Precipitação (PRECTOTCORR) e Insolação estimada para qualquer coordenada do Brasil.
    Aplica correção topoclimática de altitude (gradiente adiabático de -0.65°C / 100m)
    entre o modelo digital de terreno da grade MERRA-2 (50 km) e a altitude real do município.
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,T2M_RANGE,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,CLOUD_AMT"
        f"&community=SB&longitude={lon}&latitude={lat}&format=JSON"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    dados = response.json()
    parametros = dados["properties"]["parameter"]
    elev_grid = float(dados.get("geometry", {}).get("coordinates", [lon, lat, 500])[2])
    alt_efetiva = int(round(alt_municipio)) if alt_municipio is not None and alt_municipio > 0 else int(round(elev_grid))

    # Gradiente térmico adiabático: -0.65°C para cada 100 metros de elevação
    delta_alt = alt_efetiva - elev_grid
    lapse_corr = -0.0065 * delta_alt

    meses_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    tmax_list = []
    tmin_list = []
    insolacao_estimada = []
    for m, d in zip(meses_keys, dias_mes):
        t2m = float(parametros["T2M"][m]) + lapse_corr
        tmax_abs = float(parametros["T2M_MAX"][m]) + lapse_corr
        tmin_abs = float(parametros["T2M_MIN"][m]) + lapse_corr
        trange = float(parametros.get("T2M_RANGE", {}).get(m, 10.5))
        if trange <= 0:
            trange = max(7.5, (tmax_abs - tmin_abs) * 0.48)

        base_max = t2m + trange / 2.0
        hot_afternoon_max = (
            base_max * 0.58 + tmax_abs * 0.42
            if base_max > 27.0
            else base_max * 0.80 + tmax_abs * 0.20
        )
        tmax_list.append(round(hot_afternoon_max, 1))
        tmin_list.append(round(t2m - trange / 2.0, 1))

        rad = float(parametros.get("ALLSKY_SFC_SW_DWN", {}).get(m, 5.2))
        insolacao_estimada.append(int(round(max(95.0, min(310.0, rad * d * 1.25)))))

    rec_max = max(round(float(parametros["T2M_MAX"][m]) + lapse_corr, 1) for m in meses_keys)
    rec_min = min(round(float(parametros["T2M_MIN"][m]) + lapse_corr, 1) for m in meses_keys)

    return {
        "altitude_m": alt_efetiva,
        "altitude_grade_m": int(round(elev_grid)),
        "correcao_altitude_c": round(lapse_corr, 2),
        "tmax": tmax_list,
        "tmin": tmin_list,
        "ur":   [round(float(parametros["RH2M"][m]), 1) for m in meses_keys],
        # PRECTOTCORR vem em mm/dia; multiplicamos pelos dias do mês
        "chuva": [round(float(parametros["PRECTOTCORR"][m]) * d, 1) for m, d in zip(meses_keys, dias_mes)],
        "insolacao": insolacao_estimada,
        "nebulosidade": [round(float(parametros.get("CLOUD_AMT", {}).get(m, 50.0)), 1) for m in meses_keys],
        "recorde_tmax_abs_c": rec_max,
        "recorde_tmin_abs_c": rec_min,
    }


def main() -> None:
    from processador_clima_inmet import eh_cidade_litoranea
    parser = argparse.ArgumentParser(
        description="Busca dados climatológicos NASA POWER para qualquer coordenada do Brasil."
    )
    parser.add_argument("--nome", type=str, required=True, help="Nome do município")
    parser.add_argument("--uf", type=str, required=True, help="Sigla da UF (ex: MG, SP, BA)")
    parser.add_argument("--regiao", type=str, default="Sudeste", help="Região geográfica")
    parser.add_argument("--lat", type=float, required=True, help="Latitude decimal (ex: -18.2494)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude decimal (ex: -43.6003)")
    parser.add_argument("--alt", type=int, default=None, help="Altitude real do município em metros (opcional, ativa downscaling topoclimático)")
    parser.add_argument("--salvar", action="store_true", help="Adiciona o município ao arquivo GeoJSON")
    args = parser.parse_args()

    print(f"Consultando NASA POWER Climatology para {args.nome}-{args.uf} ({args.lat}, {args.lon}, Alt={args.alt}m)...")
    clima = buscar_clima_municipio_nasa(args.lat, args.lon, alt_municipio=args.alt)
    is_lit = eh_cidade_litoranea(args.nome, args.uf, clima["altitude_m"])
    meses_seca = sum(1 for u, c in zip(clima["ur"], clima["chuva"]) if c < 60.0 or u < 59.0)
    meses_umidos = sum(1 for u, c in zip(clima["ur"], clima["chuva"]) if u > 78.0 or c > 200.0)
    fator_persistencia_seca = 1.0 + max(0, meses_seca - 2) * 0.20
    fator_persistencia_umidade = 1.0 + max(0, meses_umidos - 4) * 0.12

    notas = []
    for m in range(12):
        av = avaliar_mes(
            clima["tmax"][m],
            clima["tmin"][m],
            clima["ur"][m],
            clima["chuva"][m],
            pm25=8.0,
            nebulosidade=clima["nebulosidade"][m],
            is_litoral=is_lit,
            mes_idx=m,
            fator_persistencia_seca=fator_persistencia_seca,
            fator_persistencia_umidade=fator_persistencia_umidade,
        )
        notas.append(av["nota_mes"])

    nota_anual = round(sum(notas) / 12.0, 1)
    print(f"\nResultado ICCB v2.6 para {args.nome}-{args.uf}:")
    print(f"  Nota Anual: {nota_anual} ({classificar_nota(nota_anual)})")
    print(f"  Altitude estimada: {clima['altitude_m']} m")
    print(f"  Litoral: {'Sim' if is_lit else 'Não'}")
    print(f"  Extremos de Reanálise: Máx {clima['recorde_tmax_abs_c']}°C | Mín {clima['recorde_tmin_abs_c']}°C")
    for idx, mes in enumerate(MESES_ORDEM):
        print(
            f"  {mes}: Nota={notas[idx]:4.1f} | Tmax={clima['tmax'][idx]:4.1f}°C | "
            f"Tmin={clima['tmin'][idx]:4.1f}°C | UR={clima['ur'][idx]:4.1f}% | Chuva={clima['chuva'][idx]:5.1f}mm | Neb={clima['nebulosidade'][idx]:4.1f}%"
        )


if __name__ == "__main__":
    main()
