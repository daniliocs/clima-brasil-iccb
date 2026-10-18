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


def buscar_clima_municipio_nasa(lat: float, lon: float) -> Dict[str, Any]:
    """
    Consulta a API gratuita NASA POWER Climatology para obter médias mensais
    de Temperatura Máxima (T2M_MAX), Mínima (T2M_MIN), Umidade Relativa (RH2M),
    Precipitação (PRECTOTCORR) e Insolação estimada para qualquer coordenada do Brasil.
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"
        f"&community=SB&longitude={lon}&latitude={lat}&format=JSON"
    )
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    dados = response.json()
    parametros = dados["properties"]["parameter"]
    elev = dados.get("geometry", {}).get("coordinates", [lon, lat, 500])[2]

    meses_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    insolacao_estimada = []
    for m, d in zip(meses_keys, dias_mes):
        rad = parametros.get("ALLSKY_SFC_SW_DWN", {}).get(m, 5.2)
        insolacao_estimada.append(int(round(max(95.0, min(310.0, rad * d * 1.25)))))

    return {
        "altitude_m": int(round(float(elev))),
        "tmax": [round(float(parametros["T2M_MAX"][m]), 1) for m in meses_keys],
        "tmin": [round(float(parametros["T2M_MIN"][m]), 1) for m in meses_keys],
        "ur":   [round(float(parametros["RH2M"][m]), 1) for m in meses_keys],
        # PRECTOTCORR vem em mm/dia; multiplicamos pelos dias do mês
        "chuva": [round(float(parametros["PRECTOTCORR"][m]) * d, 1) for m, d in zip(meses_keys, dias_mes)],
        "insolacao": insolacao_estimada,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Busca dados climatológicos NASA POWER para qualquer coordenada do Brasil."
    )
    parser.add_argument("--nome", type=str, required=True, help="Nome do município")
    parser.add_argument("--uf", type=str, required=True, help="Sigla da UF (ex: MG, SP, BA)")
    parser.add_argument("--regiao", type=str, default="Sudeste", help="Região geográfica")
    parser.add_argument("--lat", type=float, required=True, help="Latitude decimal (ex: -18.2494)")
    parser.add_argument("--lon", type=float, required=True, help="Longitude decimal (ex: -43.6003)")
    parser.add_argument("--salvar", action="store_true", help="Adiciona o município ao arquivo GeoJSON")
    args = parser.parse_args()

    print(f"Consultando NASA POWER Climatology para {args.nome}-{args.uf} ({args.lat}, {args.lon})...")
    clima = buscar_clima_municipio_nasa(args.lat, args.lon)

    notas = []
    for m in range(12):
        av = avaliar_mes(clima["tmax"][m], clima["tmin"][m], clima["ur"][m], clima["chuva"][m])
        notas.append(av["nota_mes"])

    nota_anual = round(sum(notas) / 12.0, 1)
    print(f"\nResultado ICCB v1.0 para {args.nome}-{args.uf}:")
    print(f"  Nota Anual: {nota_anual} ({classificar_nota(nota_anual)})")
    print(f"  Altitude estimada: {clima['altitude_m']} m")
    for idx, mes in enumerate(MESES_ORDEM):
        print(
            f"  {mes}: Nota={notas[idx]:4.1f} | Tmax={clima['tmax'][idx]:4.1f}°C | "
            f"Tmin={clima['tmin'][idx]:4.1f}°C | UR={clima['ur'][idx]:4.1f}% | Chuva={clima['chuva'][idx]:5.1f}mm"
        )


if __name__ == "__main__":
    main()
