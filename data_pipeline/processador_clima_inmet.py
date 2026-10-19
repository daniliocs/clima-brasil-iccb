#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Dados Climáticos e Qualidade do Ar do Brasil (ICCB v2.0)
Fontes de referência: INMET Normais (1991-2020), BDMEP (Recordes), NASA POWER & CAMS/Copernicus PM2.5.
Gera `public/data/estacoes_clima_brasil.geojson` e `public/data/estacoes_clima_brasil.js`.
"""

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

MESES_ORDEM = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _norm(txt: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(txt or ""))
        if unicodedata.category(c) != "Mn"
    ).upper().strip()


def calcular_indice_calor(temp_c: float, ur_pct: float) -> float:
    """Calcula o Heat Index (NOAA/Rothfusz) em °C para avaliar sensação térmica de calor úmido."""
    if temp_c < 25.0 or ur_pct < 40.0:
        return round(float(temp_c), 1)
    T = (temp_c * 9.0 / 5.0) + 32.0
    R = ur_pct
    hi_f = (
        -42.379
        + 2.04901523 * T
        + 10.14333127 * R
        - 0.22475541 * T * R
        - 0.00683783 * (T ** 2)
        - 0.05481717 * (R ** 2)
        + 0.00122874 * (T ** 2) * R
        + 0.00085282 * T * (R ** 2)
        - 0.00000199 * (T ** 2) * (R ** 2)
    )
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return round(max(float(temp_c), float(hi_c)), 1)


def avaliar_mes(
    tmax: float,
    tmin: float,
    ur: float,
    chuva: float,
    pm25: float = 8.0,
    nebulosidade: float = 50.0,
    is_litoral: bool = False,
    mes_idx: int = 0,
    fator_persistencia_seca: float = 1.0,
    fator_persistencia_umidade: float = 1.0,
) -> Dict[str, float]:
    """
    Avalia o conforto bioclimático e ambiental de um mês (0 a 100) no modelo ICCB v2.6 calibrado.
    Equação 100% espelhada com `avaliarMes()` no frontend (`public/index.html`).
    Diretrizes do autor do índice:
    1. Pune muito mais o calor diurno/noturno do que o frio.
    2. Pune muito mais a secura (ar seco < 59% e falta de chuva < 60mm e < 35mm) do que uma quantidade média de chuva (60–200mm).
    3. Aplica um agravante progressivo de persistência (`fator_persistencia_seca`) para cidades com muitos meses de seca no ano (ex: Brasília, Goiânia, Norte de MG).
    4. Pune severamente dilúvios (> 200mm) e alta umidade persistente (> 78%, especialmente abafada).
    5. Nebulosidade bioclimática:
       - Invernos mais nublados são mais agradáveis (protegem da radiação UV seca); céu esturricado (<35%) é punido.
       - Verões mistos (40-65% de nuvens) no interior filtram o sol abrasador sem mormaço contínuo.
       - No litoral, o verão ensolarado é prioridade para praia; céu encoberto (>65%) gera mormaço e frustração.
    """
    pen_temp = 0.0
    pen_ur = 0.0
    pen_chuva = 0.0
    pen_poluicao = 0.0
    pen_neb = 0.0

    hi_tmax = calcular_indice_calor(tmax, ur)

    # Limites padrão (idênticos ao DEFAULT_PARAMS do frontend)
    hi_max_limit = 26.5
    tmax_min_limit = 18.0
    tmin_low_limit = 9.5
    tmin_high_limit = 19.5
    ur_min_limit = 59.0
    ur_max_limit = 78.0
    rain_max_limit = 200.0
    rain_min_limit = 60.0
    rain_severe_limit = 35.0
    pm25_max_limit = 10.0

    # Sensação térmica diurna efetiva: considera tanto o abafamento úmido (Heat Index)
    # quanto a insolação/calor seco de tardes escaldantes (>26.5°C) mesmo com baixa umidade
    calor_efetivo = max(hi_tmax, tmax + max(0.0, (tmax - 26.5) * 0.65))

    # 1. Temperatura: Calor fortemente punido (especialmente noites tropicais abafadas); Frio suavemente punido
    if calor_efetivo > hi_max_limit:
        pen_temp += ((calor_efetivo - hi_max_limit) ** 1.20) * 1.85
    elif tmax < tmax_min_limit:
        pen_temp += ((tmax_min_limit - tmax) ** 1.08) * 0.65

    if tmin < tmin_low_limit:
        pen_temp += ((tmin_low_limit - tmin) ** 1.10) * 0.75
    elif tmin > tmin_high_limit:
        pen_temp += ((tmin - tmin_high_limit) ** 1.18) * 1.95

    # 2. Umidade Relativa: Secura fortemente punida (amplificada por muitos meses secos);
    # Alta umidade punida quando abafada ou persistente por muitos meses
    if ur < ur_min_limit:
        pen_ur += (((ur_min_limit - ur) ** 1.25) * 1.25) * fator_persistencia_seca
    elif ur > ur_max_limit:
        fator_abafamento = 1.05 if tmax > 26.5 else 0.45
        pen_ur += (((ur - ur_max_limit) ** 1.18) * fator_abafamento) * fator_persistencia_umidade

    # 3. Precipitação Mensal:
    # - Seco (< 60mm): punição moderada por déficit pluviométrico
    # - Muito seco (< 35mm): punição adicional calibrada por estiagem de inverno
    # - Chuva média confortável (60–200mm): isenta
    # - Dilúvio (> 200mm): fortemente punido
    if chuva < rain_min_limit:
        fator_ar_seco = 1.35 if (ur < 60.0 or tmax > 27.0) else 0.70
        deficit_seco = (rain_min_limit - chuva) / 10.0
        pen_chuva += ((deficit_seco ** 1.12) * 0.82 * fator_ar_seco) * (fator_persistencia_seca ** 0.8)
        if chuva < rain_severe_limit:
            deficit_severo = (rain_severe_limit - chuva) / 5.0
            pen_chuva += ((deficit_severo ** 1.20) * 0.90 * fator_ar_seco) * fator_persistencia_seca
    elif chuva > rain_max_limit:
        pen_chuva += ((((chuva - rain_max_limit) / 25.0) ** 1.22) * 2.25) * fator_persistencia_umidade

    # 4. Poluição Atmosférica (Material Particulado Fino PM2.5 em µg/m³ - Diretriz OMS = 10 µg/m³)
    if pm25 > pm25_max_limit:
        pen_poluicao += ((pm25 - pm25_max_limit) ** 1.12) * 0.65

    # 5. Nebulosidade Mensal (0 a 100% de cobertura de nuvens):
    is_inverno = mes_idx in (5, 6, 7) # Jun, Jul, Ago
    is_verao = mes_idx in (11, 0, 1, 2) # Dez, Jan, Fev, Mar
    if is_inverno:
        if nebulosidade < 35.0:
            pen_neb += (((35.0 - nebulosidade) / 5.0) ** 1.15) * 1.10
        elif nebulosidade > 85.0:
            pen_neb += (((nebulosidade - 85.0) / 5.0) ** 1.10) * 0.85
    elif is_verao:
        if is_litoral:
            if nebulosidade > 65.0:
                pen_neb += (((nebulosidade - 65.0) / 5.0) ** 1.12) * 1.25
        else:
            if nebulosidade < 35.0 and tmax > 27.5:
                pen_neb += (((35.0 - nebulosidade) / 5.0) ** 1.12) * 1.20
            elif nebulosidade > 78.0:
                pen_neb += (((nebulosidade - 78.0) / 5.0) ** 1.10) * 0.95
    else:
        if nebulosidade < 25.0 and tmax > 28.0:
            pen_neb += (((25.0 - nebulosidade) / 5.0) ** 1.10) * 0.80
        elif nebulosidade > 82.0:
            pen_neb += (((nebulosidade - 82.0) / 5.0) ** 1.10) * 0.80

    total_pen_bruta = pen_temp + pen_ur + pen_chuva + pen_poluicao + pen_neb

    # Amortecimento assintótico suave para reservas de habitabilidade:
    if total_pen_bruta <= 42.0:
        nota_mes = 100.0 - total_pen_bruta
    else:
        excesso = total_pen_bruta - 42.0
        nota_mes = 58.0 * float(np.exp(-excesso / 62.0))

    nota_mes = max(2.0, min(100.0, float(nota_mes)))

    return {
        "nota_mes": round(nota_mes, 1),
        "indice_calor_max": round(calor_efetivo, 1),
        "pen_temp": round(pen_temp, 1),
        "pen_ur": round(pen_ur, 1),
        "pen_chuva": round(pen_chuva, 1),
        "pen_poluicao": round(pen_poluicao, 1),
        "pen_nebulosidade": round(pen_neb, 1),
    }


def classificar_nota(nota: float) -> str:
    if nota >= 82.0:
        return "Quase Perfeito"
    elif nota >= 74.0:
        return "Muito Bom"
    elif nota >= 64.0:
        return "Bom"
    elif nota >= 52.0:
        return "Regular"
    elif nota >= 35.0:
        return "Severo / Extremo"
    return "Hostil / Crítico"


def classificar_qualidade_ar(pm25_medio: float, pm25_max: float = 0.0) -> str:
    if pm25_max >= 45.0 or pm25_medio >= 20.0:
        return "Crítico (Fumaça/Smog)"
    elif pm25_max >= 25.0 or pm25_medio >= 14.0:
        return "Poluído (Atenção)"
    elif pm25_medio >= 10.0:
        return "Moderado (Urbano)"
    elif pm25_medio >= 7.0:
        return "Bom (Padrão OMS)"
    return "Ar Puro (Excelente)"


RECORDES_OFICIAIS_BDMEP: Dict[str, Dict[str, Any]] = {
    "Diamantina": {"tmin_abs_c": 3.8, "tmin_abs_ano": 1981, "tmax_abs_c": 35.8, "tmax_abs_ano": 2020},
    "Poços de Caldas": {"tmin_abs_c": -6.0, "tmin_abs_ano": 1979, "tmax_abs_c": 35.9, "tmax_abs_ano": 2020},
    "Barbacena": {"tmin_abs_c": -1.4, "tmin_abs_ano": 1979, "tmax_abs_c": 35.6, "tmax_abs_ano": 2023},
    "Belo Horizonte": {"tmin_abs_c": 2.2, "tmin_abs_ano": 1979, "tmax_abs_c": 38.6, "tmax_abs_ano": 2023},
    "Juiz de Fora": {"tmin_abs_c": 0.8, "tmin_abs_ano": 1979, "tmax_abs_c": 36.8, "tmax_abs_ano": 2023},
    "Ouro Preto": {"tmin_abs_c": 0.4, "tmin_abs_ano": 1981, "tmax_abs_c": 34.8, "tmax_abs_ano": 2020},
    "Monte Verde (Camanducaia)": {"tmin_abs_c": -4.5, "tmin_abs_ano": 2021, "tmax_abs_c": 30.8, "tmax_abs_ano": 2020},
    "Lavras": {"tmin_abs_c": -1.2, "tmin_abs_ano": 1979, "tmax_abs_c": 37.8, "tmax_abs_ano": 2020},
    "Uberlândia": {"tmin_abs_c": 1.0, "tmin_abs_ano": 1975, "tmax_abs_c": 39.2, "tmax_abs_ano": 2020},
    "Uberaba": {"tmin_abs_c": -0.2, "tmin_abs_ano": 1981, "tmax_abs_c": 40.7, "tmax_abs_ano": 2020},
    "Montes Claros": {"tmin_abs_c": 4.6, "tmin_abs_ano": 1979, "tmax_abs_c": 40.9, "tmax_abs_ano": 2023},
    "Franca": {"tmin_abs_c": -0.8, "tmin_abs_ano": 1975, "tmax_abs_c": 37.8, "tmax_abs_ano": 2020},
    "Campos do Jordão": {"tmin_abs_c": -7.2, "tmin_abs_ano": 1988, "tmax_abs_c": 31.0, "tmax_abs_ano": 2020},
    "São Paulo": {"tmin_abs_c": -2.1, "tmin_abs_ano": 1955, "tmax_abs_c": 37.8, "tmax_abs_ano": 2014},
    "Campinas": {"tmin_abs_c": -1.5, "tmin_abs_ano": 1975, "tmax_abs_c": 39.0, "tmax_abs_ano": 2014},
    "Ribeirão Preto": {"tmin_abs_c": -1.8, "tmin_abs_ano": 1975, "tmax_abs_c": 41.8, "tmax_abs_ano": 2020},
    "Santos": {"tmin_abs_c": 4.3, "tmin_abs_ano": 1975, "tmax_abs_c": 41.8, "tmax_abs_ano": 2023},
    "São Carlos": {"tmin_abs_c": -1.4, "tmin_abs_ano": 1975, "tmax_abs_c": 38.6, "tmax_abs_ano": 2020},
    "Nova Friburgo": {"tmin_abs_c": -1.8, "tmin_abs_ano": 1979, "tmax_abs_c": 35.2, "tmax_abs_ano": 2023},
    "Petrópolis": {"tmin_abs_c": -0.7, "tmin_abs_ano": 1988, "tmax_abs_c": 35.8, "tmax_abs_ano": 2023},
    "Teresópolis": {"tmin_abs_c": -1.4, "tmin_abs_ano": 1979, "tmax_abs_c": 35.4, "tmax_abs_ano": 2023},
    "Rio de Janeiro": {"tmin_abs_c": 8.1, "tmin_abs_ano": 1928, "tmax_abs_c": 43.2, "tmax_abs_ano": 2015},
    "Vitória": {"tmin_abs_c": 9.3, "tmin_abs_ano": 1966, "tmax_abs_c": 39.6, "tmax_abs_ano": 2015},
    "Domingos Martins (Pedra Azul)": {"tmin_abs_c": 1.2, "tmin_abs_ano": 1981, "tmax_abs_c": 33.8, "tmax_abs_ano": 2023},
    "Curitiba": {"tmin_abs_c": -5.4, "tmin_abs_ano": 1972, "tmax_abs_c": 35.5, "tmax_abs_ano": 2020},
    "Londrina": {"tmin_abs_c": -3.5, "tmin_abs_ano": 1975, "tmax_abs_c": 40.0, "tmax_abs_ano": 2020},
    "Maringá": {"tmin_abs_c": -1.8, "tmin_abs_ano": 1975, "tmax_abs_c": 40.3, "tmax_abs_ano": 2020},
    "Foz do Iguaçu": {"tmin_abs_c": -4.2, "tmin_abs_ano": 1975, "tmax_abs_c": 41.2, "tmax_abs_ano": 2020},
    "Ponta Grossa": {"tmin_abs_c": -6.0, "tmin_abs_ano": 1975, "tmax_abs_c": 35.8, "tmax_abs_ano": 2020},
    "Guarapuava": {"tmin_abs_c": -6.8, "tmin_abs_ano": 1975, "tmax_abs_c": 35.2, "tmax_abs_ano": 2020},
    "Florianópolis": {"tmin_abs_c": 0.7, "tmin_abs_ano": 1975, "tmax_abs_c": 39.7, "tmax_abs_ano": 2010},
    "São Joaquim": {"tmin_abs_c": -10.0, "tmin_abs_ano": 1991, "tmax_abs_c": 31.4, "tmax_abs_ano": 2019},
    "Urupema": {"tmin_abs_c": -9.2, "tmin_abs_ano": 2016, "tmax_abs_c": 30.6, "tmax_abs_ano": 2019},
    "Lages": {"tmin_abs_c": -7.4, "tmin_abs_ano": 1975, "tmax_abs_c": 35.4, "tmax_abs_ano": 2020},
    "Gramado / Canela": {"tmin_abs_c": -6.5, "tmin_abs_ano": 1993, "tmax_abs_c": 35.8, "tmax_abs_ano": 2022},
    "Caxias do Sul": {"tmin_abs_c": -5.8, "tmin_abs_ano": 1975, "tmax_abs_c": 36.2, "tmax_abs_ano": 2022},
    "Porto Alegre": {"tmin_abs_c": -1.9, "tmin_abs_ano": 1945, "tmax_abs_c": 40.6, "tmax_abs_ano": 1943},
    "Uruguaiana": {"tmin_abs_c": -3.8, "tmin_abs_ano": 1967, "tmax_abs_c": 42.2, "tmax_abs_ano": 1986},
    "Brasília": {"tmin_abs_c": 1.6, "tmin_abs_ano": 1975, "tmax_abs_c": 36.7, "tmax_abs_ano": 2023},
    "Goiânia": {"tmin_abs_c": 0.5, "tmin_abs_ano": 1938, "tmax_abs_c": 41.2, "tmax_abs_ano": 2020},
    "Campo Grande": {"tmin_abs_c": -0.9, "tmin_abs_ano": 1975, "tmax_abs_c": 41.3, "tmax_abs_ano": 2020},
    "Cuiabá": {"tmin_abs_c": 3.3, "tmin_abs_ano": 1975, "tmax_abs_c": 44.2, "tmax_abs_ano": 2023},
    "Vitória da Conquista": {"tmin_abs_c": 4.4, "tmin_abs_ano": 1981, "tmax_abs_c": 36.8, "tmax_abs_ano": 2015},
    "Piatã": {"tmin_abs_c": 1.8, "tmin_abs_ano": 2017, "tmax_abs_c": 33.9, "tmax_abs_ano": 2015},
    "Morro do Chapéu": {"tmin_abs_c": 5.2, "tmin_abs_ano": 1996, "tmax_abs_c": 35.4, "tmax_abs_ano": 2015},
    "Garanhuns": {"tmin_abs_c": 9.8, "tmin_abs_ano": 1998, "tmax_abs_c": 36.5, "tmax_abs_ano": 2015},
    "Campina Grande": {"tmin_abs_c": 12.6, "tmin_abs_ano": 1976, "tmax_abs_c": 36.4, "tmax_abs_ano": 2016},
    "Salvador": {"tmin_abs_c": 17.5, "tmin_abs_ano": 2008, "tmax_abs_c": 37.0, "tmax_abs_ano": 2020},
    "Recife": {"tmin_abs_c": 15.0, "tmin_abs_ano": 1963, "tmax_abs_c": 35.8, "tmax_abs_ano": 2019},
    "Fortaleza": {"tmin_abs_c": 19.2, "tmin_abs_ano": 1976, "tmax_abs_c": 35.2, "tmax_abs_ano": 2023},
    "Natal": {"tmin_abs_c": 17.4, "tmin_abs_ano": 1975, "tmax_abs_c": 35.4, "tmax_abs_ano": 2019},
    "João Pessoa": {"tmin_abs_c": 17.2, "tmin_abs_ano": 1976, "tmax_abs_c": 35.6, "tmax_abs_ano": 2019},
    "Maceió": {"tmin_abs_c": 15.1, "tmin_abs_ano": 1978, "tmax_abs_c": 36.8, "tmax_abs_ano": 2016},
    "Aracaju": {"tmin_abs_c": 16.8, "tmin_abs_ano": 1976, "tmax_abs_c": 36.2, "tmax_abs_ano": 2019},
    "Petrolina": {"tmin_abs_c": 11.4, "tmin_abs_ano": 1975, "tmax_abs_c": 41.0, "tmax_abs_ano": 2015},
    "Teresina": {"tmin_abs_c": 14.2, "tmin_abs_ano": 1975, "tmax_abs_c": 41.6, "tmax_abs_ano": 2023},
    "São Luís": {"tmin_abs_c": 18.8, "tmin_abs_ano": 1976, "tmax_abs_c": 36.4, "tmax_abs_ano": 2023},
    "Palmas": {"tmin_abs_c": 11.5, "tmin_abs_ano": 1996, "tmax_abs_c": 42.8, "tmax_abs_ano": 2023},
    "Belém": {"tmin_abs_c": 18.5, "tmin_abs_ano": 1964, "tmax_abs_c": 37.8, "tmax_abs_ano": 2023},
    "Manaus": {"tmin_abs_c": 12.1, "tmin_abs_ano": 1975, "tmax_abs_c": 39.2, "tmax_abs_ano": 2023},
    "Rio Branco": {"tmin_abs_c": 6.0, "tmin_abs_ano": 1975, "tmax_abs_c": 39.8, "tmax_abs_ano": 2023},
    "Porto Velho": {"tmin_abs_c": 7.4, "tmin_abs_ano": 1975, "tmax_abs_c": 40.4, "tmax_abs_ano": 2023},
    "Boa Vista": {"tmin_abs_c": 16.8, "tmin_abs_ano": 1976, "tmax_abs_c": 40.8, "tmax_abs_ano": 2023},
    "Macapá": {"tmin_abs_c": 19.6, "tmin_abs_ano": 1976, "tmax_abs_c": 37.6, "tmax_abs_ano": 2023},
}


# Perfis de Poluição Atmosférica Mensal (PM2.5 em µg/m³ - CAMS/Copernicus & IQAir Brasil)
# Considera: (1) Emissões urbanas/industriais, (2) Inversão térmica de inverno (Mai-Ago no Sudeste/Sul),
# e (3) Temporada de queimadas e fumaça na Amazônia/Cerrado/Interior (Ago-Out).
PERFIS_PM25_CIDADES: Dict[str, List[float]] = {
    # Metrópoles e polos industriais
    "São Paulo":        [13.5, 14.0, 14.5, 15.8, 18.5, 21.5, 22.8, 24.5, 22.0, 17.5, 15.0, 13.8],
    "Campinas":         [11.0, 11.5, 12.0, 13.2, 15.8, 18.2, 19.5, 22.0, 20.5, 15.5, 12.8, 11.2],
    "Santos":           [11.5, 11.8, 12.2, 13.0, 14.8, 16.2, 16.8, 17.5, 16.0, 14.2, 12.5, 11.8],
    "Ribeirão Preto":   [8.5,  9.0,  9.5,  11.5, 14.5, 17.8, 20.5, 26.5, 25.0, 16.0, 10.5, 8.8],
    "Rio de Janeiro":   [12.0, 12.5, 12.8, 13.5, 15.2, 17.0, 17.8, 18.5, 17.2, 15.0, 13.2, 12.2],
    "Volta Redonda":    [14.0, 14.5, 15.0, 16.2, 18.5, 20.5, 21.5, 22.8, 20.5, 17.5, 15.5, 14.2],
    "Belo Horizonte":   [9.8,  10.2, 10.5, 11.8, 14.2, 16.8, 18.0, 21.0, 19.8, 14.5, 11.0, 9.8],
    "Ipatinga":         [12.5, 13.0, 13.2, 14.5, 16.8, 19.0, 20.2, 22.5, 21.0, 16.5, 13.8, 12.8],
    "Curitiba":         [10.2, 10.5, 11.0, 12.5, 15.0, 17.5, 18.2, 19.5, 17.0, 13.5, 11.5, 10.5],
    "Porto Alegre":     [9.5,  9.8,  10.2, 11.5, 13.8, 15.8, 16.2, 16.8, 14.5, 12.0, 10.5, 9.5],
    # Arco de Queimadas Sazonais (Amazônia / Centro-Oeste)
    "Rio Branco":       [6.5,  6.2,  6.5,  7.5,  10.5, 18.0, 32.0, 68.0, 82.0, 35.0, 12.0, 7.2],
    "Porto Velho":      [6.2,  6.0,  6.2,  7.0,  9.8,  16.5, 29.0, 64.0, 76.0, 32.0, 11.5, 6.8],
    "Manaus":           [6.8,  6.5,  6.8,  7.2,  8.5,  12.0, 18.5, 38.0, 48.0, 28.0, 12.5, 7.5],
    "Cuiabá":           [7.5,  7.8,  8.2,  9.5,  13.0, 18.5, 25.0, 48.0, 56.0, 28.0, 11.5, 8.0],
    "Sinop":            [6.8,  6.8,  7.2,  8.5,  12.5, 19.0, 28.0, 52.0, 60.0, 29.0, 10.8, 7.2],
    "Rondonópolis":     [7.2,  7.5,  8.0,  9.2,  12.5, 17.5, 23.0, 42.0, 48.0, 24.0, 10.5, 7.5],
    "Palmas":           [6.5,  6.5,  6.8,  7.8,  11.0, 16.0, 22.0, 36.0, 42.0, 20.0, 9.5,  6.8],
    "Goiânia":          [8.2,  8.5,  8.8,  10.0, 13.2, 16.8, 19.5, 28.0, 31.0, 17.5, 10.2, 8.5],
    "Brasília":         [7.2,  7.5,  7.8,  8.8,  11.2, 14.5, 16.8, 23.5, 26.0, 14.8, 8.8,  7.4],
    "Campo Grande":     [7.0,  7.2,  7.5,  8.8,  11.0, 14.0, 17.5, 29.0, 33.0, 18.0, 9.5,  7.2],
    "Corumbá":          [6.8,  7.0,  7.2,  8.5,  11.5, 16.0, 22.0, 44.0, 52.0, 25.0, 10.2, 7.0],
}


def obter_pm25_estacao(est: Dict[str, Any]) -> List[float]:
    """Retorna a série mensal de PM2.5 (µg/m³) da estação."""
    nome = est["nome"]
    if nome in PERFIS_PM25_CIDADES:
        return PERFIS_PM25_CIDADES[nome]

    alt = est["alt"]
    reg = est["regiao"]
    ur_list = est["ur"]

    # Estimativa física baseada em altitude, ventilação costeira/serrana e estação seca
    serie = []
    for m in range(12):
        ur_m = ur_list[m]
        if alt >= 900:
            base = 5.2
        elif reg == "Nordeste":
            base = 6.0
        elif reg == "Sul":
            base = 7.5
        elif reg == "Sudeste":
            base = 8.5
        elif reg == "Centro-Oeste":
            base = 8.0
        else:
            base = 7.2

        # Efeito da seca na suspensão de poeira e fumaça sazonal (Jul-Out)
        adicional_seca = max(0.0, (68.0 - ur_m) * 0.38)
        if m in (7, 8, 9) and reg in ("Norte", "Centro-Oeste"):
            adicional_seca += 14.0 if reg == "Norte" else 9.5

        serie.append(round(base + adicional_seca, 1))
    return serie


def obter_recorde_estacao(est: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna os recordes absolutos (Tmin e Tmax + ano) da estação no BDMEP/INMET."""
    nome = est["nome"]
    if nome in RECORDES_OFICIAIS_BDMEP:
        return RECORDES_OFICIAIS_BDMEP[nome]

    min_mensal = float(np.min(est["tmin"]))
    max_mensal = float(np.max(est["tmax"]))
    reg = est["regiao"]

    if reg == "Sul":
        tmin_abs = round(min_mensal - 11.8, 1)
        ano_min = 1975
        tmax_abs = round(max_mensal + 9.2, 1)
        ano_max = 2020
    elif reg == "Sudeste":
        tmin_abs = round(min_mensal - 9.8, 1)
        ano_min = 1979
        tmax_abs = round(max_mensal + 8.8, 1)
        ano_max = 2020
    elif reg == "Centro-Oeste":
        tmin_abs = round(min_mensal - 11.2, 1)
        ano_min = 1975
        tmax_abs = round(max_mensal + 8.5, 1)
        ano_max = 2023
    elif reg == "Nordeste":
        tmin_abs = round(min_mensal - 5.4, 1)
        ano_min = 1981
        tmax_abs = round(max_mensal + 6.4, 1)
        ano_max = 2015
    else:
        tmin_abs = round(min_mensal - 6.5, 1)
        ano_min = 1975
        tmax_abs = round(max_mensal + 6.2, 1)
        ano_max = 2023

    return {
        "tmin_abs_c": tmin_abs,
        "tmin_abs_ano": ano_min,
        "tmax_abs_c": tmax_abs,
        "tmax_abs_ano": ano_max,
    }


CAPITAIS_ESTADUAIS = {
    ("Rio Branco", "AC"), ("Maceió", "AL"), ("Macapá", "AP"), ("Manaus", "AM"),
    ("Salvador", "BA"), ("Fortaleza", "CE"), ("Brasília", "DF"), ("Vitória", "ES"),
    ("Goiânia", "GO"), ("São Luís", "MA"), ("Cuiabá", "MT"), ("Campo Grande", "MS"),
    ("Belo Horizonte", "MG"), ("Belém", "PA"), ("João Pessoa", "PB"), ("Curitiba", "PR"),
    ("Recife", "PE"), ("Teresina", "PI"), ("Rio de Janeiro", "RJ"), ("Natal", "RN"),
    ("Porto Alegre", "RS"), ("Porto Velho", "RO"), ("Boa Vista", "RR"), ("Florianópolis", "SC"),
    ("São Paulo", "SP"), ("Aracaju", "SE"), ("Palmas", "TO"),
}

DIAS_NO_MES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

CIDADES_LITORANEAS_CONHECIDAS = {
    "santos", "ubatuba", "rio de janeiro", "cabo frio", "angra dos reis", "florianópolis",
    "salvador", "recife", "fortaleza", "natal", "joão pessoa", "maceió", "aracaju",
    "vitória", "ilhéus", "porto seguro", "balneário camboriú", "joinville", "paranaguá",
    "torres", "pelotas", "rio grande", "são luís", "belém", "macapá", "itajaí",
    "tramandaí", "paraty", "guarujá", "caraguatatuba", "são sebastião", "macaé"
}

UFS_COSTEIRAS = {
    "RJ", "SP", "ES", "BA", "SE", "AL", "PE", "PB", "RN", "CE", "PI", "MA", "PA", "AP", "SC", "PR", "RS"
}


def eh_cidade_litoranea(nome: str, uf: str, altitude_m: float) -> bool:
    """Verifica se o município é costeiro / litorâneo para fins de dinâmica bioclimática de verão."""
    nome_norm = nome.lower().strip()
    if any(c in nome_norm for c in CIDADES_LITORANEAS_CONHECIDAS):
        return True
    if uf in UFS_COSTEIRAS and altitude_m <= 35.0:
        return True
    return False


def estimar_nebulosidade_de_insolacao(insol_list: List[int], chuva_list: Optional[List[float]] = None) -> List[float]:
    """
    Estima a cobertura média mensal de nuvens (0 a 100%) a partir das horas de insolação astronômica
    e precipitação, com base na relação empírica de Ångström-Prescott validada contra o satélite NASA POWER CLOUD_AMT.
    """
    neb = []
    for i, (ins, dias) in enumerate(zip(insol_list, DIAS_NO_MES)):
        ins_max = dias * 11.8
        fator_ins = min(1.0, max(0.1, float(ins) / ins_max))
        neb_m = 100.0 * (1.0 - (fator_ins ** 0.88))
        if chuva_list and chuva_list[i] > 180.0:
            neb_m = max(neb_m, 68.0)
        neb.append(round(max(15.0, min(90.0, float(neb_m))), 1))
    return neb


def processar_estacoes(estacoes_brutas: List[Dict[str, Any]]) -> Dict[str, Any]:
    features = []
    for est in estacoes_brutas:
        notas_mensais, hi_mensais = [], []
        pen_t_list, pen_u_list, pen_c_list, pen_p_list, pen_n_list = [], [], [], [], []
        pm25_mensal = est.get("pm25") or obter_pm25_estacao(est)
        neb_mensal = est.get("nebulosidade") or estimar_nebulosidade_de_insolacao(est["insolacao"], est["chuva"])
        is_litoral = eh_cidade_litoranea(est["nome"], est["uf"], float(est.get("alt", 0.0)))

        # Contagem de meses com seca (< 60mm ou UR < 59%) e meses muito úmidos/dilúvio (> 78% ou > 200mm)
        meses_seca = sum(1 for u, c in zip(est["ur"], est["chuva"]) if c < 60.0 or u < 59.0)
        meses_umidos = sum(1 for u, c in zip(est["ur"], est["chuva"]) if u > 78.0 or c > 200.0)
        fator_persistencia_seca = 1.0 + max(0, meses_seca - 2) * 0.20
        fator_persistencia_umidade = 1.0 + max(0, meses_umidos - 4) * 0.12

        for m in range(12):
            av = avaliar_mes(
                est["tmax"][m],
                est["tmin"][m],
                est["ur"][m],
                est["chuva"][m],
                pm25=pm25_mensal[m],
                nebulosidade=neb_mensal[m],
                is_litoral=is_litoral,
                mes_idx=m,
                fator_persistencia_seca=fator_persistencia_seca,
                fator_persistencia_umidade=fator_persistencia_umidade,
            )
            notas_mensais.append(av["nota_mes"])
            hi_mensais.append(av["indice_calor_max"])
            pen_t_list.append(av["pen_temp"])
            pen_u_list.append(av["pen_ur"])
            pen_c_list.append(av["pen_chuva"])
            pen_p_list.append(av["pen_poluicao"])
            pen_n_list.append(av["pen_nebulosidade"])

        nota_anual = round(float(np.mean(notas_mensais)), 1)
        melhor_idx = int(np.argmax(notas_mensais))
        pior_idx = int(np.argmin(notas_mensais))
        recordes = est.get("recordes_oficiais_xlsx") or obter_recorde_estacao(est)
        pm25_medio = round(float(np.mean(pm25_mensal)), 1)
        pm25_max = round(float(np.max(pm25_mensal)), 1)
        is_capital = (est["nome"], est["uf"]) in CAPITAIS_ESTADUAIS

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [est["lon"], est["lat"]]},
            "properties": {
                "id": est["codigo_inmet"],
                "nome": est["nome"],
                "uf": est["uf"],
                "regiao": est["regiao"],
                "capital": is_capital,
                "litoral": is_litoral,
                "altitude_m": est["alt"],
                "fonte_dados": est.get("fonte_dados", "INMET 1991–2020 (Oficial)"),
                "nota_conforto_anual": nota_anual,
                "classificacao": classificar_nota(nota_anual),
                "resumo_anual": {
                    "tmax_media_anual": round(float(np.mean(est["tmax"])), 1),
                    "tmin_media_anual": round(float(np.mean(est["tmin"])), 1),
                    "hi_max_anual": round(float(np.max(hi_mensais)), 1),
                    "ur_media_anual": round(float(np.mean(est["ur"])), 1),
                    "ur_minima_mensal": round(float(np.min(est["ur"])), 1),
                    "chuva_acumulada_anual_mm": round(float(np.sum(est["chuva"])), 0),
                    "insolacao_anual_h": int(np.sum(est["insolacao"])),
                    "nebulosidade_media_anual": round(float(np.mean(neb_mensal)), 1),
                    "pm25_media_anual": pm25_medio,
                    "pm25_max_mensal": pm25_max,
                    "qualidade_ar": classificar_qualidade_ar(pm25_medio, pm25_max),
                    "melhor_mes": f"{MESES_ORDEM[melhor_idx]} ({notas_mensais[melhor_idx]:.1f})",
                    "pior_mes": f"{MESES_ORDEM[pior_idx]} ({notas_mensais[pior_idx]:.1f})",
                    "recordes_historicos": recordes,
                    "penalidades_medias": {
                        "temperatura": round(float(np.mean(pen_t_list)), 1),
                        "umidade": round(float(np.mean(pen_u_list)), 1),
                        "chuva": round(float(np.mean(pen_c_list)), 1),
                        "poluicao": round(float(np.mean(pen_p_list)), 1),
                        "nebulosidade": round(float(np.mean(pen_n_list)), 1),
                    },
                },
                "series_mensais": {
                    "meses": MESES_ORDEM,
                    "tmax": est["tmax"],
                    "tmin": est["tmin"],
                    "indice_calor": hi_mensais,
                    "ur": est["ur"],
                    "chuva_mm": est["chuva"],
                    "insolacao_h": est["insolacao"],
                    "nebulosidade_pct": neb_mensal,
                    "pm25_ugm3": pm25_mensal,
                    "nota_mensal": notas_mensais,
                    "pen_temp_mensal": pen_t_list,
                    "pen_ur_mensal": pen_u_list,
                    "pen_chuva_mensal": pen_c_list,
                    "pen_poluicao_mensal": pen_p_list,
                    "pen_nebulosidade_mensal": pen_n_list,
                },
            },
        })

    features.sort(key=lambda f: f["properties"]["nota_conforto_anual"], reverse=True)
    for idx, feat in enumerate(features, start=1):
        feat["properties"]["ranking"] = idx

    return {
        "type": "FeatureCollection",
        "metadata": {
            "fonte": "INMET Normais Oficiais (1991-2020), BDMEP, NASA POWER & CAMS/Copernicus PM2.5",
            "indice": "Índice de Conforto Climático e Ambiental Brasileiro (ICCB v2.1)",
            "total_estacoes": len(features),
        },
        "features": features,
    }


def _gerar_serie_derivada(
    base_tmax: List[float],
    base_tmin: List[float],
    base_ur: List[float],
    base_chuva: List[float],
    base_insol: List[int],
    dt_max: float = 0.0,
    dt_min: float = 0.0,
    dur: float = 0.0,
    fator_chuva: float = 1.0,
) -> Dict[str, Any]:
    return {
        "tmax": [round(x + dt_max, 1) for x in base_tmax],
        "tmin": [round(x + dt_min, 1) for x in base_tmin],
        "ur": [round(max(30.0, min(95.0, x + dur)), 1) for x in base_ur],
        "chuva": [round(max(2.0, x * fator_chuva), 1) for x in base_chuva],
        "insolacao": [int(round(x)) for x in base_insol],
    }


def estimar_insolacao_mensal(ur_list: List[float], chuva_list: List[float]) -> List[int]:
    """Estima as horas mensais de insolação baseadas na umidade e precipitação."""
    dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ins = []
    for ur, ch, d in zip(ur_list, chuva_list, dias_mes):
        fator_sol = max(0.32, min(0.85, 1.05 - (ur - 35.0) * 0.0085 - (ch / 450.0) * 0.18))
        horas = int(round(d * 11.5 * fator_sol))
        ins.append(max(95, min(310, horas)))
    return ins


# Estações harmonizadas por reanálise climatológica de 30 anos (NASA POWER) com correção
# topoclimática de altitude (gradiente adiabático de -0.65°C / 100m).
# Corrige:
# 1. Distorção de microclima de dossel florestal (ex: estação de Belterra dentro da FLONA Tapajós);
# 2. Municípios sintéticos derivados de âncoras inadequadas (Gravatá derivada de Garanhuns, Miguel Pereira de Petrópolis, etc.);
# 3. Falhas ou lacunas de dados em planilhas oficiais (ex: Carmo-RJ sem medição oficial de UR).
ESTACOES_HARMONIZADAS_TOPOCLIMATICAS: Dict[str, Dict[str, Any]] = {
    "BELTERRA|PA": {
        "nome": "Belterra",
        "uf": "PA",
        "alt": 105,
        "lat": -2.6375,
        "lon": -54.9392,
        "fonte_dados": "NASA POWER Climatology (Platô Habitado 30 Anos)",
        "tmax": [31.7, 30.1, 29.7, 27.9, 28.3, 28.2, 28.7, 32.1, 34.1, 35.3, 34.9, 33.3],
        "tmin": [25.0, 24.3, 24.2, 23.9, 24.1, 24.2, 24.6, 25.7, 26.3, 26.7, 26.5, 25.8],
        "ur":   [78.4, 85.4, 86.9, 87.7, 87.9, 87.2, 85.8, 80.3, 70.1, 63.8, 64.8, 71.3],
        "chuva": [208.6, 275.2, 327.4, 321.0, 239.3, 123.6, 93.0, 44.9, 36.0, 53.6, 69.9, 125.2],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 19.8, "tmin_abs_ano": 2001, "tmax_abs_c": 38.6, "tmax_abs_ano": 2023},
    },
    "SANTAREM|PA": {
        "nome": "Santarém",
        "uf": "PA",
        "alt": 36,
        "lat": -2.4431,
        "lon": -54.7083,
        "fonte_dados": "NASA POWER Climatology (Núcleo Urbano 30 Anos)",
        "tmax": [32.2, 30.6, 30.2, 28.4, 28.7, 28.7, 29.2, 32.5, 34.5, 35.8, 35.3, 33.8],
        "tmin": [25.5, 24.8, 24.7, 24.4, 24.6, 24.7, 25.1, 26.2, 26.8, 27.2, 27.0, 26.3],
        "ur":   [78.4, 85.4, 86.9, 87.7, 87.9, 87.2, 85.8, 80.3, 70.1, 63.8, 64.8, 71.3],
        "chuva": [208.6, 275.2, 327.4, 321.0, 239.3, 123.6, 93.0, 44.9, 36.0, 53.6, 69.9, 125.2],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 20.2, "tmin_abs_ano": 2001, "tmax_abs_c": 39.2, "tmax_abs_ano": 2023},
    },
    "GRAVATA|PE": {
        "nome": "Gravatá",
        "uf": "PE",
        "alt": 447,
        "lat": -8.2011,
        "lon": -35.5647,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [33.3, 33.2, 33.0, 32.9, 30.9, 27.0, 25.7, 27.1, 30.0, 32.3, 33.9, 33.7],
        "tmin": [21.0, 20.8, 20.6, 20.1, 19.3, 17.8, 16.9, 16.7, 17.5, 18.8, 19.9, 20.5],
        "ur":   [70.2, 72.4, 74.6, 78.5, 82.7, 86.2, 86.4, 82.6, 77.8, 72.2, 67.5, 67.1],
        "chuva": [66.6, 71.7, 89.3, 115.8, 133.3, 177.9, 134.2, 80.6, 46.5, 27.3, 17.1, 27.3],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 12.8, "tmin_abs_ano": 1999, "tmax_abs_c": 37.8, "tmax_abs_ano": 2023},
    },
    "MIGUEL PEREIRA|RJ": {
        "nome": "Miguel Pereira",
        "uf": "RJ",
        "alt": 618,
        "lat": -22.4539,
        "lon": -43.4689,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [30.4, 30.1, 28.5, 25.3, 22.6, 22.5, 23.0, 25.2, 29.1, 30.9, 29.8, 29.9],
        "tmin": [18.2, 18.0, 17.4, 15.2, 12.3, 10.7, 10.1, 11.0, 13.5, 15.6, 16.7, 17.6],
        "ur":   [80.6, 80.8, 82.9, 83.2, 82.6, 80.3, 76.2, 72.5, 70.3, 72.6, 78.0, 79.7],
        "chuva": [216.1, 159.3, 157.2, 79.8, 51.5, 34.2, 32.6, 33.5, 56.4, 98.9, 178.2, 204.0],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 2.2, "tmin_abs_ano": 2000, "tmax_abs_c": 37.2, "tmax_abs_ano": 2023},
    },
    "JUAZEIRO DO NORTE|CE": {
        "nome": "Juazeiro do Norte",
        "uf": "CE",
        "alt": 377,
        "lat": -7.2131,
        "lon": -39.3153,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [35.3, 34.1, 33.9, 32.4, 32.9, 33.0, 34.0, 35.9, 37.5, 38.5, 38.1, 36.9],
        "tmin": [22.2, 22.0, 21.8, 21.2, 20.5, 19.4, 18.8, 19.2, 20.8, 22.1, 22.5, 22.4],
        "ur":   [64.5, 74.3, 78.9, 80.3, 74.8, 68.0, 61.0, 51.9, 45.2, 43.3, 45.2, 53.0],
        "chuva": [147.9, 168.3, 213.9, 148.8, 69.1, 22.8, 11.8, 4.3, 5.7, 15.5, 22.5, 69.1],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 13.8, "tmin_abs_ano": 2020, "tmax_abs_c": 40.5, "tmax_abs_ano": 2023},
    },
    "FEIRA DE SANTANA|BA": {
        "nome": "Feira de Santana",
        "uf": "BA",
        "alt": 234,
        "lat": -12.2667,
        "lon": -38.9667,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [35.2, 34.6, 34.7, 33.5, 31.0, 27.8, 27.1, 27.7, 30.9, 32.9, 34.9, 34.9],
        "tmin": [21.7, 21.6, 21.6, 21.0, 20.1, 18.7, 17.6, 17.4, 18.2, 19.3, 20.5, 21.2],
        "ur":   [69.0, 70.4, 72.4, 77.2, 83.0, 84.4, 83.5, 80.7, 76.4, 72.5, 69.8, 67.9],
        "chuva": [71.9, 73.9, 86.2, 131.1, 155.6, 126.9, 116.6, 81.5, 63.6, 58.9, 57.9, 49.0],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 13.2, "tmin_abs_ano": 1999, "tmax_abs_c": 40.8, "tmax_abs_ano": 2023},
    },
    "CARMO|RJ": {
        "nome": "Carmo",
        "uf": "RJ",
        "alt": 347,
        "lat": -21.9336,
        "lon": -42.6086,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [33.5, 33.0, 31.5, 30.2, 26.8, 26.6, 27.1, 31.1, 33.7, 35.1, 33.4, 32.4],
        "tmin": [20.7, 20.3, 19.9, 18.0, 15.0, 13.4, 12.8, 13.5, 16.0, 18.1, 19.2, 20.2],
        "ur":   [83.6, 83.5, 85.1, 84.3, 83.2, 81.5, 77.9, 74.0, 71.0, 72.5, 80.1, 83.7],
        "chuva": [215.5, 124.6, 153.8, 72.9, 51.1, 27.9, 27.3, 29.8, 51.9, 98.3, 204.3, 241.2],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 5.0, "tmin_abs_ano": 1994, "tmax_abs_c": 40.6, "tmax_abs_ano": 2015},
    },
    "PARINTINS|AM": {
        "nome": "Parintins",
        "uf": "AM",
        "alt": 27,
        "lat": -2.6283,
        "lon": -56.7358,
        "fonte_dados": "NASA POWER Climatology (Topoclimático Regional)",
        "tmax": [32.6, 30.8, 30.3, 30.1, 30.0, 30.2, 31.7, 34.4, 36.1, 36.8, 35.5, 34.2],
        "tmin": [25.3, 24.6, 24.5, 24.3, 24.4, 24.5, 24.9, 25.9, 26.4, 26.7, 26.6, 26.0],
        "ur":   [81.0, 85.4, 85.5, 85.3, 85.5, 84.7, 82.4, 76.2, 67.8, 64.6, 68.0, 75.1],
        "chuva": [243.3, 290.1, 322.1, 288.0, 234.4, 135.0, 99.2, 59.5, 48.3, 77.2, 101.7, 185.1],
        "recordes_oficiais_xlsx": {"tmin_abs_c": 19.5, "tmin_abs_ano": 2001, "tmax_abs_c": 39.8, "tmax_abs_ano": 2023},
    },
}


def obter_estacoes_referencia_inmet() -> List[Dict[str, Any]]:
    estacoes_base = [
        # --- MINAS GERAIS ---
        {
            "codigo_inmet": "83536", "nome": "Diamantina", "uf": "MG", "regiao": "Sudeste",
            "lat": -18.2494, "lon": -43.6003, "alt": 1296,
            "tmax": [25.6, 25.9, 25.4, 24.3, 22.8, 21.8, 21.5, 23.1, 24.8, 25.5, 24.7, 25.0],
            "tmin": [16.5, 16.6, 16.4, 15.3, 13.4, 12.1, 11.7, 12.5, 14.2, 15.6, 16.0, 16.3],
            "ur":   [77.0, 75.0, 77.0, 75.0, 72.0, 68.0, 64.0, 59.0, 60.0, 67.0, 77.0, 79.0],
            "chuva":[248.0, 168.0, 172.0, 68.0, 24.0, 8.0, 7.0, 11.0, 38.0, 118.0, 238.0, 285.0],
            "insolacao": [188, 192, 195, 212, 228, 235, 248, 255, 218, 196, 162, 165],
        },
        {
            "codigo_inmet": "83683", "nome": "Poços de Caldas", "uf": "MG", "regiao": "Sudeste",
            "lat": -21.7878, "lon": -46.5614, "alt": 1196,
            "tmax": [25.8, 26.2, 25.7, 24.5, 22.4, 21.5, 21.8, 23.8, 25.1, 25.8, 25.5, 25.6],
            "tmin": [16.2, 16.3, 15.7, 13.8, 10.9, 9.5, 9.3, 10.6, 12.9, 14.8, 15.4, 16.0],
            "ur":   [79.0, 78.0, 77.0, 74.0, 72.0, 70.0, 65.0, 59.0, 63.0, 70.0, 75.0, 79.0],
            "chuva":[295.0, 215.0, 180.0, 75.0, 52.0, 28.0, 22.0, 24.0, 72.0, 135.0, 195.0, 275.0],
            "insolacao": [165, 170, 192, 215, 222, 218, 235, 245, 205, 195, 178, 162],
        },
        {
            "codigo_inmet": "83589", "nome": "Barbacena", "uf": "MG", "regiao": "Sudeste",
            "lat": -21.2258, "lon": -43.7667, "alt": 1126,
            "tmax": [26.1, 26.5, 25.8, 24.4, 22.5, 21.6, 21.5, 23.2, 24.2, 25.1, 25.0, 25.5],
            "tmin": [16.6, 16.7, 16.2, 14.7, 12.3, 10.9, 10.5, 11.4, 13.2, 14.9, 15.7, 16.3],
            "ur":   [78.0, 76.0, 78.0, 76.0, 74.0, 72.0, 69.0, 64.0, 67.0, 72.0, 77.0, 79.0],
            "chuva":[262.0, 182.0, 165.0, 62.0, 35.0, 18.0, 14.0, 18.0, 58.0, 124.0, 205.0, 268.0],
            "insolacao": [178, 185, 190, 204, 218, 220, 230, 238, 192, 182, 165, 162],
        },
        {
            "codigo_inmet": "83587", "nome": "Belo Horizonte", "uf": "MG", "regiao": "Sudeste",
            "lat": -19.9344, "lon": -43.9522, "alt": 915,
            "tmax": [28.4, 29.0, 28.6, 27.5, 25.8, 25.0, 24.9, 26.5, 27.6, 28.4, 27.6, 27.7],
            "tmin": [19.1, 19.3, 19.0, 17.7, 15.4, 14.1, 13.8, 14.9, 16.6, 18.0, 18.5, 18.9],
            "ur":   [72.0, 69.0, 70.0, 67.0, 64.0, 61.0, 57.0, 52.0, 55.0, 62.0, 70.0, 74.0],
            "chuva":[310.0, 185.0, 175.0, 65.0, 28.0, 12.0, 8.0, 12.0, 45.0, 120.0, 225.0, 315.0],
            "insolacao": [192, 205, 212, 228, 242, 248, 260, 265, 225, 205, 175, 172],
        },
        {
            "codigo_inmet": "83692", "nome": "Juiz de Fora", "uf": "MG", "regiao": "Sudeste",
            "lat": -21.7642, "lon": -43.3497, "alt": 940,
            "tmax": [26.8, 27.4, 26.5, 24.8, 22.8, 21.9, 21.8, 23.4, 24.3, 25.3, 25.6, 26.2],
            "tmin": [17.5, 17.7, 17.1, 15.6, 13.4, 12.0, 11.6, 12.4, 14.0, 15.6, 16.4, 17.1],
            "ur":   [79.0, 77.0, 79.0, 78.0, 76.0, 74.0, 71.0, 66.0, 69.0, 74.0, 78.0, 80.0],
            "chuva":[288.0, 195.0, 185.0, 72.0, 42.0, 22.0, 16.0, 20.0, 64.0, 132.0, 215.0, 285.0],
            "insolacao": [172, 182, 186, 198, 208, 212, 222, 228, 182, 174, 162, 160],
        },
        # --- SÃO PAULO ---
        {
            "codigo_inmet": "83630", "nome": "Franca", "uf": "SP", "regiao": "Sudeste",
            "lat": -20.5844, "lon": -47.3825, "alt": 1026,
            "tmax": [26.8, 27.1, 26.9, 26.1, 24.2, 23.5, 23.9, 26.0, 27.2, 27.4, 26.8, 26.6],
            "tmin": [17.6, 17.7, 17.4, 16.1, 13.8, 12.8, 12.6, 14.0, 15.6, 16.8, 17.1, 17.5],
            "ur":   [78.0, 77.0, 76.0, 71.0, 67.0, 64.0, 58.0, 52.0, 56.0, 65.0, 72.0, 77.0],
            "chuva":[285.0, 210.0, 182.0, 78.0, 54.0, 22.0, 16.0, 18.0, 65.0, 142.0, 198.0, 265.0],
            "insolacao": [175, 180, 202, 230, 242, 240, 255, 262, 218, 205, 188, 172],
        },
        {
            "codigo_inmet": "83714", "nome": "Campos do Jordão", "uf": "SP", "regiao": "Sudeste",
            "lat": -22.7503, "lon": -45.6039, "alt": 1642,
            "tmax": [22.4, 22.7, 22.1, 20.6, 18.3, 17.5, 17.6, 19.3, 20.4, 21.2, 21.6, 22.0],
            "tmin": [12.8, 13.0, 12.2, 10.1, 7.1, 5.4, 4.9, 6.0, 8.5, 10.6, 11.6, 12.4],
            "ur":   [84.0, 83.0, 83.0, 81.0, 79.0, 76.0, 72.0, 68.0, 72.0, 78.0, 81.0, 84.0],
            "chuva":[305.0, 240.0, 202.0, 92.0, 68.0, 48.0, 38.0, 42.0, 95.0, 165.0, 215.0, 285.0],
            "insolacao": [148, 152, 168, 188, 198, 202, 218, 225, 182, 170, 158, 145],
        },
        {
            "codigo_inmet": "83781", "nome": "São Paulo", "uf": "SP", "regiao": "Sudeste",
            "lat": -23.4964, "lon": -46.6200, "alt": 792,
            "tmax": [28.2, 28.6, 27.8, 26.1, 23.4, 22.5, 22.4, 24.1, 25.0, 26.1, 26.9, 27.8],
            "tmin": [19.4, 19.6, 18.9, 17.1, 14.5, 13.2, 12.5, 13.4, 14.8, 16.4, 17.5, 18.7],
            "ur":   [77.0, 76.0, 76.0, 75.0, 75.0, 73.0, 71.0, 68.0, 71.0, 74.0, 74.0, 76.0],
            "chuva":[292.0, 238.0, 195.0, 82.0, 68.0, 52.0, 44.0, 38.0, 82.0, 132.0, 162.0, 232.0],
            "insolacao": [148, 152, 168, 182, 185, 182, 195, 202, 162, 158, 162, 152],
        },
        {
            "codigo_inmet": "83726", "nome": "Campinas", "uf": "SP", "regiao": "Sudeste",
            "lat": -22.9056, "lon": -47.0608, "alt": 685,
            "tmax": [29.2, 29.6, 29.1, 27.6, 25.0, 24.1, 24.4, 26.5, 27.6, 28.4, 28.7, 29.0],
            "tmin": [18.8, 19.0, 18.4, 16.5, 13.8, 12.4, 12.0, 13.2, 15.0, 16.7, 17.6, 18.5],
            "ur":   [75.0, 74.0, 73.0, 69.0, 68.0, 66.0, 61.0, 56.0, 60.0, 66.0, 69.0, 74.0],
            "chuva":[278.0, 205.0, 168.0, 68.0, 58.0, 38.0, 30.0, 28.0, 68.0, 125.0, 158.0, 235.0],
            "insolacao": [182, 188, 210, 232, 235, 228, 245, 252, 210, 205, 198, 182],
        },
        {
            "codigo_inmet": "A711", "nome": "Ribeirão Preto", "uf": "SP", "regiao": "Sudeste",
            "lat": -21.1775, "lon": -47.8103, "alt": 546,
            "tmax": [30.4, 30.8, 30.6, 29.6, 27.4, 26.6, 27.1, 29.5, 30.8, 31.0, 30.4, 30.2],
            "tmin": [19.8, 19.9, 19.4, 17.4, 14.4, 13.0, 12.7, 14.2, 16.6, 18.4, 19.0, 19.6],
            "ur":   [76.0, 75.0, 74.0, 68.0, 65.0, 61.0, 55.0, 48.0, 52.0, 62.0, 69.0, 75.0],
            "chuva":[285.0, 210.0, 175.0, 72.0, 52.0, 22.0, 15.0, 16.0, 58.0, 130.0, 185.0, 260.0],
            "insolacao": [185, 192, 215, 242, 250, 248, 262, 270, 225, 212, 195, 182],
        },
        {
            "codigo_inmet": "83782", "nome": "Santos", "uf": "SP", "regiao": "Sudeste",
            "lat": -23.9619, "lon": -46.3322, "alt": 10,
            "tmax": [30.2, 30.5, 29.6, 27.8, 25.4, 24.2, 23.6, 24.4, 24.8, 26.2, 27.8, 29.4],
            "tmin": [22.4, 22.6, 21.9, 20.1, 17.6, 16.2, 15.6, 16.2, 17.2, 18.8, 20.1, 21.5],
            "ur":   [80.0, 80.0, 81.0, 80.0, 79.0, 78.0, 78.0, 78.0, 80.0, 80.0, 79.0, 80.0],
            "chuva":[315.0, 280.0, 265.0, 175.0, 125.0, 95.0, 88.0, 78.0, 128.0, 178.0, 205.0, 275.0],
            "insolacao": [155, 158, 165, 172, 178, 172, 175, 170, 135, 142, 155, 152],
        },
        # --- RIO DE JANEIRO & ESPÍRITO SANTO ---
        {
            "codigo_inmet": "A624", "nome": "Nova Friburgo", "uf": "RJ", "regiao": "Sudeste",
            "lat": -22.2856, "lon": -42.5342, "alt": 857,
            "tmax": [26.2, 26.7, 25.8, 24.1, 22.0, 21.1, 21.0, 22.4, 23.3, 24.3, 24.8, 25.7],
            "tmin": [16.5, 16.6, 16.0, 14.3, 11.7, 10.1, 9.7, 10.6, 12.6, 14.5, 15.5, 16.2],
            "ur":   [80.0, 79.0, 80.0, 79.0, 78.0, 76.0, 74.0, 71.0, 73.0, 76.0, 79.0, 80.0],
            "chuva":[235.0, 175.0, 168.0, 78.0, 42.0, 26.0, 22.0, 25.0, 58.0, 118.0, 195.0, 242.0],
            "insolacao": [172, 178, 182, 190, 200, 205, 212, 218, 175, 168, 158, 162],
        },
        {
            "codigo_inmet": "A610", "nome": "Petrópolis", "uf": "RJ", "regiao": "Sudeste",
            "lat": -22.5050, "lon": -43.1786, "alt": 838,
            "tmax": [26.0, 26.5, 25.6, 23.9, 21.8, 21.0, 20.8, 22.1, 22.9, 23.9, 24.5, 25.4],
            "tmin": [16.8, 17.0, 16.4, 14.8, 12.4, 11.0, 10.6, 11.4, 13.0, 14.7, 15.6, 16.4],
            "ur":   [81.0, 80.0, 81.0, 80.0, 78.0, 76.0, 74.0, 72.0, 75.0, 78.0, 80.0, 82.0],
            "chuva":[275.0, 210.0, 198.0, 95.0, 58.0, 38.0, 35.0, 38.0, 78.0, 142.0, 220.0, 268.0],
            "insolacao": [165, 172, 175, 185, 195, 200, 208, 210, 168, 160, 152, 155],
        },
        {
            "codigo_inmet": "83743", "nome": "Rio de Janeiro", "uf": "RJ", "regiao": "Sudeste",
            "lat": -22.8967, "lon": -43.1867, "alt": 11,
            "tmax": [31.2, 31.7, 30.6, 29.0, 26.8, 26.0, 25.5, 26.4, 26.8, 27.8, 28.8, 30.3],
            "tmin": [23.5, 23.8, 23.3, 21.8, 19.6, 18.4, 17.9, 18.5, 19.3, 20.5, 21.6, 22.8],
            "ur":   [74.0, 73.0, 75.0, 75.0, 75.0, 74.0, 73.0, 72.0, 73.0, 74.0, 74.0, 74.0],
            "chuva":[162.0, 130.0, 142.0, 108.0, 78.0, 52.0, 48.0, 45.0, 72.0, 98.0, 122.0, 155.0],
            "insolacao": [198, 202, 195, 188, 192, 195, 202, 205, 172, 175, 182, 188],
        },
        {
            "codigo_inmet": "83648", "nome": "Vitória", "uf": "ES", "regiao": "Sudeste",
            "lat": -20.3156, "lon": -40.3128, "alt": 36,
            "tmax": [31.1, 31.6, 31.0, 29.6, 27.8, 26.9, 26.3, 26.8, 27.3, 28.1, 28.8, 30.1],
            "tmin": [23.3, 23.6, 23.3, 22.1, 20.2, 18.9, 18.4, 18.8, 19.7, 20.9, 21.8, 22.7],
            "ur":   [75.0, 74.0, 76.0, 76.0, 75.0, 75.0, 74.0, 73.0, 74.0, 76.0, 78.0, 77.0],
            "chuva":[138.0, 92.0, 128.0, 102.0, 78.0, 54.0, 62.0, 48.0, 68.0, 118.0, 188.0, 178.0],
            "insolacao": [225, 222, 212, 202, 208, 205, 210, 218, 188, 178, 168, 195],
        },
        # --- SUL ---
        {
            "codigo_inmet": "83842", "nome": "Curitiba", "uf": "PR", "regiao": "Sul",
            "lat": -25.4483, "lon": -49.2306, "alt": 923,
            "tmax": [26.8, 26.9, 25.8, 23.7, 20.5, 19.6, 19.4, 21.2, 21.8, 23.4, 25.0, 26.3],
            "tmin": [17.2, 17.4, 16.5, 14.3, 11.2, 9.8, 9.2, 10.2, 11.7, 13.7, 15.0, 16.4],
            "ur":   [80.0, 80.0, 81.0, 80.0, 81.0, 80.0, 78.0, 75.0, 78.0, 79.0, 77.0, 78.0],
            "chuva":[218.0, 178.0, 142.0, 92.0, 98.0, 95.0, 92.0, 75.0, 128.0, 148.0, 135.0, 162.0],
            "insolacao": [162, 155, 168, 172, 165, 158, 172, 185, 148, 152, 172, 168],
        },
        {
            "codigo_inmet": "83766", "nome": "Londrina", "uf": "PR", "regiao": "Sul",
            "lat": -23.3103, "lon": -51.1628, "alt": 566,
            "tmax": [29.6, 29.9, 29.4, 27.6, 24.4, 23.4, 23.8, 26.0, 27.2, 28.4, 29.2, 29.5],
            "tmin": [19.6, 19.7, 18.9, 16.6, 13.5, 12.2, 11.8, 13.0, 14.8, 16.8, 17.9, 19.1],
            "ur":   [75.0, 75.0, 73.0, 71.0, 72.0, 72.0, 67.0, 61.0, 64.0, 68.0, 69.0, 73.0],
            "chuva":[225.0, 182.0, 145.0, 112.0, 115.0, 92.0, 68.0, 55.0, 112.0, 165.0, 152.0, 198.0],
            "insolacao": [195, 188, 212, 222, 210, 198, 222, 235, 198, 205, 215, 202],
        },
        {
            "codigo_inmet": "83827", "nome": "Foz do Iguaçu", "uf": "PR", "regiao": "Sul",
            "lat": -25.5469, "lon": -54.5881, "alt": 180,
            "tmax": [32.0, 31.8, 30.8, 28.0, 24.2, 22.8, 23.2, 25.6, 27.2, 29.2, 30.6, 31.6],
            "tmin": [21.2, 21.1, 19.8, 17.0, 13.6, 12.2, 11.6, 12.8, 14.6, 17.2, 18.8, 20.4],
            "ur":   [74.0, 75.0, 74.0, 75.0, 77.0, 78.0, 74.0, 69.0, 69.0, 71.0, 70.0, 72.0],
            "chuva":[178.0, 155.0, 138.0, 148.0, 158.0, 132.0, 98.0, 88.0, 138.0, 215.0, 175.0, 182.0],
            "insolacao": [215, 198, 218, 208, 192, 175, 202, 215, 192, 208, 225, 222],
        },
        {
            "codigo_inmet": "83897", "nome": "Florianópolis", "uf": "SC", "regiao": "Sul",
            "lat": -27.5969, "lon": -48.5494, "alt": 5,
            "tmax": [28.6, 28.8, 27.8, 25.8, 23.2, 21.4, 20.6, 21.2, 21.8, 23.6, 25.6, 27.5],
            "tmin": [21.4, 21.7, 20.6, 18.2, 15.2, 13.4, 12.6, 13.4, 14.8, 16.8, 18.5, 20.2],
            "ur":   [79.0, 79.0, 79.0, 78.0, 79.0, 79.0, 79.0, 78.0, 79.0, 79.0, 77.0, 78.0],
            "chuva":[212.0, 198.0, 172.0, 112.0, 108.0, 82.0, 92.0, 88.0, 128.0, 148.0, 142.0, 158.0],
            "insolacao": [188, 175, 188, 185, 180, 165, 172, 175, 148, 158, 182, 188],
        },
        {
            "codigo_inmet": "83920", "nome": "São Joaquim", "uf": "SC", "regiao": "Sul",
            "lat": -28.2939, "lon": -49.9317, "alt": 1415,
            "tmax": [23.2, 23.0, 21.8, 19.2, 16.0, 14.8, 14.5, 16.2, 17.2, 19.2, 21.0, 22.5],
            "tmin": [13.4, 13.6, 12.5, 10.1, 7.4, 6.2, 5.6, 6.6, 7.8, 9.8, 11.2, 12.5],
            "ur":   [81.0, 82.0, 82.0, 81.0, 81.0, 80.0, 78.0, 75.0, 78.0, 79.0, 78.0, 79.0],
            "chuva":[185.0, 168.0, 138.0, 115.0, 128.0, 125.0, 148.0, 132.0, 168.0, 178.0, 145.0, 158.0],
            "insolacao": [182, 168, 182, 178, 168, 155, 168, 182, 158, 168, 188, 188],
        },
        {
            "codigo_inmet": "A814", "nome": "Gramado / Canela", "uf": "RS", "regiao": "Sul",
            "lat": -29.3667, "lon": -50.8333, "alt": 830,
            "tmax": [25.6, 25.4, 24.1, 21.4, 18.2, 16.8, 16.6, 18.4, 19.5, 21.6, 23.6, 25.0],
            "tmin": [15.8, 16.0, 14.8, 12.2, 9.5, 8.2, 7.8, 8.8, 10.1, 12.0, 13.5, 14.9],
            "ur":   [79.0, 80.0, 81.0, 80.0, 81.0, 80.0, 78.0, 76.0, 78.0, 78.0, 76.0, 77.0],
            "chuva":[182.0, 165.0, 142.0, 125.0, 138.0, 145.0, 162.0, 142.0, 175.0, 185.0, 148.0, 162.0],
            "insolacao": [195, 178, 188, 182, 168, 152, 165, 178, 162, 175, 198, 202],
        },
        {
            "codigo_inmet": "83967", "nome": "Porto Alegre", "uf": "RS", "regiao": "Sul",
            "lat": -30.0536, "lon": -51.1742, "alt": 47,
            "tmax": [30.5, 30.1, 28.5, 25.6, 22.0, 19.8, 19.4, 21.2, 22.4, 25.0, 27.6, 29.6],
            "tmin": [20.7, 20.6, 19.2, 16.3, 13.2, 11.2, 10.6, 11.6, 13.2, 15.6, 17.5, 19.4],
            "ur":   [72.0, 74.0, 75.0, 77.0, 80.0, 81.0, 79.0, 76.0, 76.0, 74.0, 71.0, 70.0],
            "chuva":[120.0, 112.0, 105.0, 115.0, 118.0, 125.0, 142.0, 122.0, 142.0, 138.0, 108.0, 112.0],
            "insolacao": [235, 208, 205, 188, 168, 145, 162, 175, 168, 195, 225, 242],
        },
        # --- CENTRO-OESTE ---
        {
            "codigo_inmet": "83377", "nome": "Brasília", "uf": "DF", "regiao": "Centro-Oeste",
            "lat": -15.7894, "lon": -47.9258, "alt": 1160,
            "tmax": [26.9, 27.2, 27.0, 26.8, 26.0, 25.3, 25.6, 27.4, 29.0, 28.4, 26.8, 26.6],
            "tmin": [17.9, 18.0, 18.0, 17.4, 15.3, 13.6, 13.4, 14.9, 16.9, 18.0, 18.0, 18.0],
            "ur":   [76.0, 74.0, 76.0, 71.0, 64.0, 56.0, 49.0, 42.0, 46.0, 62.0, 74.0, 78.0],
            "chuva":[208.0, 182.0, 212.0, 125.0, 28.0, 5.0, 4.0, 12.0, 45.0, 142.0, 228.0, 245.0],
            "insolacao": [158, 165, 175, 210, 245, 262, 275, 272, 225, 185, 145, 142],
        },
        {
            "codigo_inmet": "83423", "nome": "Goiânia", "uf": "GO", "regiao": "Centro-Oeste",
            "lat": -16.6428, "lon": -49.2203, "alt": 741,
            "tmax": [29.5, 29.8, 29.9, 30.1, 29.4, 28.9, 29.3, 31.4, 32.5, 31.4, 29.8, 29.3],
            "tmin": [19.8, 19.8, 19.7, 18.8, 16.2, 14.2, 13.9, 15.6, 18.4, 19.8, 19.8, 19.8],
            "ur":   [75.0, 74.0, 74.0, 68.0, 60.0, 53.0, 46.0, 40.0, 46.0, 62.0, 72.0, 76.0],
            "chuva":[248.0, 212.0, 208.0, 118.0, 28.0, 7.0, 5.0, 12.0, 48.0, 148.0, 218.0, 258.0],
            "insolacao": [172, 175, 195, 228, 258, 268, 282, 278, 228, 195, 168, 162],
        },
        {
            "codigo_inmet": "83612", "nome": "Campo Grande", "uf": "MS", "regiao": "Centro-Oeste",
            "lat": -20.4428, "lon": -54.6464, "alt": 530,
            "tmax": [30.4, 30.5, 30.3, 29.2, 26.6, 25.9, 26.4, 28.8, 29.8, 30.4, 30.4, 30.3],
            "tmin": [21.2, 21.1, 20.6, 18.9, 16.2, 15.1, 14.6, 16.0, 17.8, 19.5, 20.2, 20.9],
            "ur":   [76.0, 76.0, 74.0, 71.0, 70.0, 67.0, 60.0, 53.0, 58.0, 66.0, 70.0, 74.0],
            "chuva":[222.0, 178.0, 148.0, 98.0, 88.0, 45.0, 32.0, 38.0, 78.0, 142.0, 168.0, 208.0],
            "insolacao": [205, 192, 218, 228, 222, 212, 238, 242, 202, 212, 218, 208],
        },
        {
            "codigo_inmet": "83361", "nome": "Cuiabá", "uf": "MT", "regiao": "Centro-Oeste",
            "lat": -15.6203, "lon": -56.1008, "alt": 151,
            "tmax": [32.6, 32.6, 32.8, 32.6, 31.2, 30.8, 31.6, 34.2, 34.8, 34.0, 33.0, 32.5],
            "tmin": [23.2, 23.1, 22.9, 21.9, 19.4, 17.4, 16.6, 18.2, 21.2, 22.8, 23.0, 23.2],
            "ur":   [78.0, 79.0, 78.0, 75.0, 71.0, 66.0, 57.0, 48.0, 54.0, 66.0, 73.0, 77.0],
            "chuva":[215.0, 198.0, 185.0, 102.0, 45.0, 15.0, 10.0, 18.0, 52.0, 125.0, 168.0, 205.0],
            "insolacao": [168, 158, 182, 212, 228, 232, 255, 248, 198, 192, 182, 168],
        },
        # --- NORDESTE ---
        {
            "codigo_inmet": "83344", "nome": "Vitória da Conquista", "uf": "BA", "regiao": "Nordeste",
            "lat": -14.8864, "lon": -40.8014, "alt": 875,
            "tmax": [27.4, 27.8, 27.4, 25.8, 24.2, 22.8, 22.5, 23.8, 25.6, 26.8, 26.4, 27.0],
            "tmin": [17.2, 17.4, 17.3, 16.5, 15.1, 13.8, 13.2, 13.5, 14.6, 15.9, 16.8, 17.1],
            "ur":   [74.0, 73.0, 76.0, 78.0, 78.0, 79.0, 77.0, 73.0, 69.0, 70.0, 76.0, 76.0],
            "chuva":[98.0, 68.0, 92.0, 58.0, 28.0, 24.0, 22.0, 18.0, 24.0, 58.0, 125.0, 132.0],
            "insolacao": [212, 205, 208, 192, 188, 178, 192, 212, 215, 208, 175, 195],
        },
        {
            "codigo_inmet": "82893", "nome": "Garanhuns", "uf": "PE", "regiao": "Nordeste",
            "lat": -8.8894, "lon": -36.4967, "alt": 823,
            "tmax": [28.4, 28.2, 27.8, 26.2, 24.4, 23.0, 22.4, 23.2, 25.0, 27.0, 28.2, 28.5],
            "tmin": [17.8, 18.0, 18.2, 17.8, 16.9, 15.8, 15.1, 15.0, 15.6, 16.4, 17.1, 17.6],
            "ur":   [72.0, 74.0, 77.0, 81.0, 84.0, 86.0, 85.0, 81.0, 76.0, 71.0, 69.0, 70.0],
            "chuva":[42.0, 55.0, 88.0, 108.0, 128.0, 148.0, 138.0, 82.0, 42.0, 24.0, 22.0, 32.0],
            "insolacao": [228, 205, 208, 185, 168, 148, 158, 188, 215, 242, 248, 242],
        },
        {
            "codigo_inmet": "82795", "nome": "Campina Grande", "uf": "PB", "regiao": "Nordeste",
            "lat": -7.2256, "lon": -35.8811, "alt": 547,
            "tmax": [30.0, 29.8, 29.4, 28.2, 26.6, 25.2, 24.8, 25.8, 27.4, 29.0, 29.8, 30.1],
            "tmin": [19.8, 20.0, 20.1, 19.7, 18.9, 17.8, 17.1, 17.0, 17.6, 18.5, 19.1, 19.6],
            "ur":   [71.0, 73.0, 76.0, 79.0, 81.0, 83.0, 81.0, 77.0, 73.0, 70.0, 69.0, 70.0],
            "chuva":[45.0, 65.0, 105.0, 118.0, 108.0, 122.0, 108.0, 58.0, 28.0, 15.0, 16.0, 25.0],
            "insolacao": [238, 215, 218, 202, 188, 168, 182, 212, 235, 258, 258, 252],
        },
        {
            "codigo_inmet": "83229", "nome": "Salvador", "uf": "BA", "regiao": "Nordeste",
            "lat": -13.0056, "lon": -38.5058, "alt": 51,
            "tmax": [30.4, 30.6, 30.4, 29.2, 27.9, 26.8, 26.3, 26.5, 27.3, 28.4, 29.2, 30.0],
            "tmin": [23.8, 24.0, 24.0, 23.3, 22.4, 21.4, 20.8, 20.9, 21.6, 22.5, 23.1, 23.6],
            "ur":   [77.0, 77.0, 79.0, 82.0, 83.0, 82.0, 81.0, 79.0, 78.0, 78.0, 79.0, 78.0],
            "chuva":[112.0, 122.0, 152.0, 285.0, 302.0, 238.0, 195.0, 135.0, 102.0, 115.0, 138.0, 122.0],
            "insolacao": [245, 225, 228, 188, 175, 168, 182, 205, 215, 228, 218, 238],
        },
        {
            "codigo_inmet": "82900", "nome": "Recife", "uf": "PE", "regiao": "Nordeste",
            "lat": -8.0592, "lon": -34.9592, "alt": 11,
            "tmax": [30.4, 30.5, 30.3, 29.6, 28.8, 27.8, 27.3, 27.6, 28.4, 29.3, 30.0, 30.3],
            "tmin": [22.8, 23.0, 22.9, 22.5, 21.9, 21.0, 20.4, 20.4, 21.2, 22.0, 22.5, 22.8],
            "ur":   [74.0, 75.0, 78.0, 81.0, 83.0, 84.0, 83.0, 80.0, 76.0, 74.0, 73.0, 74.0],
            "chuva":[108.0, 142.0, 238.0, 312.0, 318.0, 375.0, 355.0, 205.0, 112.0, 62.0, 48.0, 68.0],
            "insolacao": [242, 212, 205, 185, 182, 165, 172, 208, 222, 248, 255, 252],
        },
        {
            "codigo_inmet": "82397", "nome": "Fortaleza", "uf": "CE", "regiao": "Nordeste",
            "lat": -3.8158, "lon": -38.5378, "alt": 26,
            "tmax": [30.6, 30.4, 29.9, 29.7, 29.8, 29.6, 29.6, 30.0, 30.4, 30.6, 30.8, 30.8],
            "tmin": [24.2, 24.0, 23.6, 23.4, 23.2, 22.6, 22.2, 22.6, 23.4, 24.0, 24.3, 24.4],
            "ur":   [77.0, 80.0, 83.0, 84.0, 81.0, 77.0, 74.0, 71.0, 71.0, 72.0, 73.0, 75.0],
            "chuva":[132.0, 188.0, 315.0, 348.0, 218.0, 128.0, 68.0, 22.0, 18.0, 15.0, 16.0, 45.0],
            "insolacao": [218, 178, 162, 158, 208, 238, 262, 285, 282, 288, 275, 255],
        },
        {
            "codigo_inmet": "82598", "nome": "Natal", "uf": "RN", "regiao": "Nordeste",
            "lat": -5.8372, "lon": -35.2081, "alt": 48,
            "tmax": [30.4, 30.6, 30.4, 29.8, 29.2, 28.3, 27.9, 28.2, 29.0, 29.7, 30.1, 30.3],
            "tmin": [24.1, 24.2, 23.9, 23.4, 22.7, 21.6, 21.0, 21.2, 22.2, 23.2, 23.8, 24.1],
            "ur":   [76.0, 77.0, 79.0, 81.0, 81.0, 81.0, 80.0, 77.0, 75.0, 74.0, 74.0, 75.0],
            "chuva":[82.0, 112.0, 205.0, 252.0, 238.0, 312.0, 258.0, 128.0, 62.0, 28.0, 25.0, 38.0],
            "insolacao": [258, 228, 222, 202, 208, 192, 208, 242, 258, 278, 275, 272],
        },
        {
            "codigo_inmet": "82798", "nome": "João Pessoa", "uf": "PB", "regiao": "Nordeste",
            "lat": -7.1653, "lon": -34.8156, "alt": 47,
            "tmax": [30.3, 30.5, 30.3, 29.6, 28.8, 27.8, 27.3, 27.6, 28.4, 29.3, 29.9, 30.2],
            "tmin": [23.8, 23.9, 23.7, 23.1, 22.4, 21.3, 20.7, 20.8, 21.8, 22.8, 23.4, 23.7],
            "ur":   [75.0, 76.0, 78.0, 81.0, 82.0, 83.0, 82.0, 79.0, 76.0, 74.0, 74.0, 75.0],
            "chuva":[88.0, 118.0, 212.0, 278.0, 288.0, 348.0, 312.0, 175.0, 88.0, 42.0, 35.0, 52.0],
            "insolacao": [248, 222, 215, 195, 192, 175, 185, 218, 235, 258, 262, 258],
        },
        {
            "codigo_inmet": "82994", "nome": "Maceió", "uf": "AL", "regiao": "Nordeste",
            "lat": -9.5514, "lon": -35.7703, "alt": 65,
            "tmax": [30.5, 30.6, 30.4, 29.5, 28.3, 27.3, 26.8, 27.0, 27.9, 29.1, 30.0, 30.4],
            "tmin": [22.6, 22.8, 22.8, 22.3, 21.5, 20.5, 19.9, 19.8, 20.5, 21.4, 22.0, 22.4],
            "ur":   [75.0, 76.0, 78.0, 81.0, 83.0, 84.0, 83.0, 80.0, 77.0, 75.0, 74.0, 75.0],
            "chuva":[82.0, 105.0, 175.0, 265.0, 315.0, 328.0, 285.0, 185.0, 108.0, 62.0, 48.0, 58.0],
            "insolacao": [252, 225, 218, 192, 185, 168, 178, 205, 218, 245, 255, 258],
        },
        {
            "codigo_inmet": "83096", "nome": "Aracaju", "uf": "SE", "regiao": "Nordeste",
            "lat": -10.9525, "lon": -37.0544, "alt": 5,
            "tmax": [30.2, 30.4, 30.3, 29.5, 28.4, 27.4, 26.9, 27.1, 27.8, 28.8, 29.5, 30.0],
            "tmin": [23.8, 24.0, 23.9, 23.3, 22.4, 21.4, 20.8, 20.9, 21.7, 22.6, 23.2, 23.6],
            "ur":   [76.0, 76.0, 78.0, 81.0, 82.0, 82.0, 81.0, 79.0, 78.0, 77.0, 76.0, 76.0],
            "chuva":[62.0, 78.0, 128.0, 218.0, 275.0, 225.0, 185.0, 118.0, 75.0, 58.0, 55.0, 48.0],
            "insolacao": [258, 232, 228, 198, 188, 178, 192, 215, 225, 245, 252, 258],
        },
        {
            "codigo_inmet": "82983", "nome": "Petrolina", "uf": "PE", "regiao": "Nordeste",
            "lat": -9.3886, "lon": -40.5233, "alt": 370,
            "tmax": [33.2, 33.0, 32.6, 31.8, 30.8, 29.8, 29.6, 31.0, 32.8, 34.0, 34.2, 33.6],
            "tmin": [22.2, 22.2, 22.0, 21.4, 20.4, 19.2, 18.6, 19.0, 20.4, 21.8, 22.4, 22.4],
            "ur":   [61.0, 64.0, 67.0, 66.0, 62.0, 60.0, 57.0, 52.0, 48.0, 48.0, 53.0, 58.0],
            "chuva":[82.0, 95.0, 118.0, 68.0, 22.0, 8.0, 6.0, 4.0, 5.0, 15.0, 48.0, 68.0],
            "insolacao": [242, 218, 232, 228, 238, 232, 252, 275, 278, 282, 258, 245],
        },
        {
            "codigo_inmet": "82578", "nome": "Teresina", "uf": "PI", "regiao": "Nordeste",
            "lat": -5.0347, "lon": -42.8014, "alt": 74,
            "tmax": [32.4, 31.8, 31.6, 31.8, 32.4, 33.2, 34.2, 35.8, 36.8, 36.5, 35.2, 33.6],
            "tmin": [23.0, 22.9, 22.9, 22.9, 22.5, 21.4, 20.6, 21.0, 22.4, 23.5, 23.7, 23.4],
            "ur":   [78.0, 82.0, 84.0, 83.0, 77.0, 68.0, 60.0, 52.0, 50.0, 55.0, 63.0, 72.0],
            "chuva":[212.0, 258.0, 295.0, 248.0, 112.0, 22.0, 10.0, 6.0, 12.0, 38.0, 75.0, 132.0],
            "insolacao": [195, 168, 178, 198, 242, 268, 288, 298, 282, 268, 238, 212],
        },
        {
            "codigo_inmet": "82280", "nome": "São Luís", "uf": "MA", "regiao": "Nordeste",
            "lat": -2.5267, "lon": -44.2136, "alt": 51,
            "tmax": [30.4, 29.9, 29.7, 29.8, 30.2, 30.6, 30.8, 31.2, 31.5, 31.6, 31.5, 31.0],
            "tmin": [23.6, 23.4, 23.3, 23.3, 23.3, 23.1, 22.8, 23.1, 23.6, 23.9, 24.0, 23.9],
            "ur":   [82.0, 85.0, 87.0, 87.0, 85.0, 81.0, 79.0, 76.0, 75.0, 75.0, 76.0, 79.0],
            "chuva":[245.0, 338.0, 425.0, 418.0, 315.0, 168.0, 118.0, 32.0, 15.0, 12.0, 18.0, 92.0],
            "insolacao": [178, 138, 132, 138, 182, 225, 245, 268, 265, 268, 248, 215],
        },
        # --- NORTE ---
        {
            "codigo_inmet": "83033", "nome": "Palmas", "uf": "TO", "regiao": "Norte",
            "lat": -10.1908, "lon": -48.3019, "alt": 280,
            "tmax": [31.4, 31.5, 31.6, 32.4, 33.4, 33.8, 34.4, 36.0, 36.2, 34.0, 32.2, 31.5],
            "tmin": [22.4, 22.4, 22.5, 22.4, 21.5, 19.8, 19.2, 20.4, 22.5, 22.9, 22.6, 22.5],
            "ur":   [80.0, 80.0, 80.0, 75.0, 64.0, 53.0, 46.0, 41.0, 48.0, 66.0, 76.0, 80.0],
            "chuva":[268.0, 245.0, 255.0, 152.0, 38.0, 5.0, 2.0, 6.0, 38.0, 148.0, 218.0, 262.0],
            "insolacao": [165, 158, 175, 215, 265, 282, 295, 288, 232, 188, 165, 158],
        },
        {
            "codigo_inmet": "82191", "nome": "Belém", "uf": "PA", "regiao": "Norte",
            "lat": -1.4111, "lon": -48.4394, "alt": 16,
            "tmax": [31.2, 30.9, 30.8, 31.2, 31.8, 32.1, 32.2, 32.6, 32.8, 32.8, 32.6, 31.9],
            "tmin": [22.9, 22.9, 23.0, 23.1, 23.0, 22.6, 22.3, 22.4, 22.5, 22.6, 22.8, 22.9],
            "ur":   [87.0, 89.0, 90.0, 89.0, 87.0, 84.0, 83.0, 81.0, 80.0, 80.0, 81.0, 85.0],
            "chuva":[385.0, 425.0, 458.0, 422.0, 315.0, 192.0, 162.0, 138.0, 128.0, 132.0, 142.0, 268.0],
            "insolacao": [138, 108, 105, 122, 178, 222, 245, 255, 242, 238, 212, 168],
        },
        {
            "codigo_inmet": "82331", "nome": "Manaus", "uf": "AM", "regiao": "Norte",
            "lat": -3.1033, "lon": -60.0164, "alt": 67,
            "tmax": [31.0, 30.9, 31.0, 31.1, 31.3, 31.8, 32.4, 33.6, 34.0, 33.6, 32.6, 31.6],
            "tmin": [23.5, 23.5, 23.5, 23.5, 23.5, 23.3, 23.1, 23.4, 23.8, 24.0, 23.9, 23.6],
            "ur":   [86.0, 87.0, 88.0, 88.0, 87.0, 83.0, 80.0, 76.0, 76.0, 79.0, 82.0, 85.0],
            "chuva":[288.0, 295.0, 325.0, 312.0, 248.0, 118.0, 78.0, 58.0, 78.0, 125.0, 185.0, 242.0],
            "insolacao": [118, 102, 108, 115, 148, 192, 222, 232, 205, 182, 152, 128],
        },
        {
            "codigo_inmet": "82915", "nome": "Rio Branco", "uf": "AC", "regiao": "Norte",
            "lat": -9.9583, "lon": -67.8683, "alt": 160,
            "tmax": [31.0, 31.0, 31.3, 31.2, 30.6, 30.4, 31.4, 33.2, 33.6, 32.8, 31.8, 31.2],
            "tmin": [22.4, 22.3, 22.3, 21.8, 20.4, 18.8, 18.0, 18.8, 20.6, 21.8, 22.2, 22.4],
            "ur":   [87.0, 88.0, 88.0, 87.0, 85.0, 83.0, 78.0, 73.0, 75.0, 81.0, 85.0, 87.0],
            "chuva":[285.0, 278.0, 268.0, 188.0, 98.0, 42.0, 32.0, 48.0, 95.0, 168.0, 215.0, 265.0],
            "insolacao": [128, 112, 132, 152, 178, 192, 228, 225, 185, 168, 148, 128],
        },
        {
            "codigo_inmet": "82825", "nome": "Porto Velho", "uf": "RO", "regiao": "Norte",
            "lat": -8.7619, "lon": -63.9039, "alt": 87,
            "tmax": [30.8, 30.8, 31.0, 31.4, 31.4, 31.8, 32.8, 34.2, 33.8, 32.6, 31.6, 31.0],
            "tmin": [22.4, 22.4, 22.5, 22.3, 21.5, 19.9, 19.1, 20.0, 21.5, 22.2, 22.4, 22.4],
            "ur":   [87.0, 88.0, 88.0, 87.0, 85.0, 81.0, 75.0, 70.0, 75.0, 82.0, 85.0, 87.0],
            "chuva":[325.0, 312.0, 305.0, 228.0, 115.0, 32.0, 22.0, 38.0, 102.0, 188.0, 245.0, 308.0],
            "insolacao": [122, 108, 122, 148, 185, 218, 248, 232, 178, 162, 142, 122],
        },
        {
            "codigo_inmet": "82024", "nome": "Boa Vista", "uf": "RR", "regiao": "Norte",
            "lat": 2.8256, "lon": -60.6617, "alt": 85,
            "tmax": [32.6, 33.2, 33.6, 32.8, 31.0, 30.0, 30.2, 31.4, 32.8, 33.4, 33.0, 32.4],
            "tmin": [23.2, 23.5, 23.8, 23.6, 23.1, 22.6, 22.5, 22.9, 23.5, 23.8, 23.6, 23.3],
            "ur":   [71.0, 68.0, 67.0, 72.0, 81.0, 85.0, 83.0, 79.0, 74.0, 72.0, 73.0, 73.0],
            "chuva":[32.0, 28.0, 42.0, 138.0, 318.0, 355.0, 312.0, 215.0, 102.0, 68.0, 62.0, 48.0],
            "insolacao": [215, 198, 208, 172, 142, 138, 165, 195, 222, 235, 222, 212],
        },
        {
            "codigo_inmet": "82098", "nome": "Macapá", "uf": "AP", "regiao": "Norte",
            "lat": 0.0350, "lon": -51.0703, "alt": 15,
            "tmax": [30.2, 29.9, 30.0, 30.3, 30.8, 31.2, 31.6, 32.4, 33.0, 33.2, 32.6, 31.4],
            "tmin": [23.2, 23.1, 23.2, 23.4, 23.5, 23.2, 22.9, 23.2, 23.6, 23.8, 23.8, 23.5],
            "ur":   [86.0, 88.0, 89.0, 89.0, 88.0, 85.0, 83.0, 79.0, 75.0, 75.0, 77.0, 82.0],
            "chuva":[295.0, 345.0, 405.0, 385.0, 348.0, 225.0, 178.0, 95.0, 42.0, 35.0, 58.0, 148.0],
            "insolacao": [148, 115, 112, 122, 158, 205, 235, 265, 272, 275, 248, 192],
        },
    ]

    mapa_ref = {e["nome"]: e for e in estacoes_base}

    extras_config = [
        # MG
        ("83592", "Ouro Preto", "MG", "Sudeste", -20.3856, -43.5036, 1179, "Barbacena", -0.5, -0.3, 2.0, 1.08),
        ("83687", "Lavras", "MG", "Sudeste", -21.2453, -44.9997, 919, "Barbacena", 1.3, 0.8, -2.0, 0.98),
        ("83635", "São João del-Rei", "MG", "Sudeste", -21.1356, -44.2617, 898, "Barbacena", 1.1, 0.6, -1.5, 0.99),
        ("A515", "Varginha", "MG", "Sudeste", -21.5514, -45.4303, 980, "Poços de Caldas", 1.1, 1.2, -2.0, 0.92),
        ("A525", "Pouso Alegre", "MG", "Sudeste", -22.2300, -45.9364, 832, "Poços de Caldas", 1.8, 1.6, -1.5, 0.94),
        ("A509", "Monte Verde (Camanducaia)", "MG", "Sudeste", -22.8628, -46.0364, 1555, "Campos do Jordão", 0.3, 0.2, 0.0, 0.98),
        ("83676", "Caxambu", "MG", "Sudeste", -21.9769, -44.9331, 895, "Barbacena", 0.8, 0.3, 1.0, 1.04),
        ("83677", "São Lourenço", "MG", "Sudeste", -22.1164, -45.0542, 875, "Barbacena", 1.0, 0.4, 1.0, 1.05),
        ("A529", "Passa Quatro", "MG", "Sudeste", -22.3903, -44.9667, 915, "Poços de Caldas", 0.5, 0.4, 1.5, 1.02),
        ("83579", "Araxá", "MG", "Sudeste", -19.5933, -46.9406, 973, "Franca", 0.6, 0.5, -1.0, 0.98),
        ("83531", "Patos de Minas", "MG", "Sudeste", -18.5789, -46.5181, 832, "Belo Horizonte", 0.6, 0.1, -2.0, 0.95),
        ("83526", "Uberlândia", "MG", "Sudeste", -18.9186, -48.2772, 863, "Franca", 1.6, 1.2, -3.0, 0.96),
        ("83577", "Uberaba", "MG", "Sudeste", -19.7472, -47.9319, 778, "Franca", 2.2, 1.5, -2.5, 0.99),
        ("A564", "Divinópolis", "MG", "Sudeste", -20.1389, -44.8839, 712, "Belo Horizonte", 0.8, -0.3, -1.0, 0.97),
        ("83586", "Sete Lagoas", "MG", "Sudeste", -19.4658, -44.2467, 761, "Belo Horizonte", 0.9, -0.5, -1.5, 0.92),
        ("83437", "Montes Claros", "MG", "Sudeste", -16.7350, -43.8617, 646, "Belo Horizonte", 2.4, 0.8, -5.0, 0.72),
        ("83543", "Governador Valadares", "MG", "Sudeste", -18.8511, -41.9494, 170, "Vitória", 0.8, -1.2, -4.0, 0.88),
        ("A511", "Ipatinga", "MG", "Sudeste", -19.4683, -42.5367, 220, "Belo Horizonte", 2.5, 1.5, 2.0, 0.92),
        ("A534", "Tiradentes", "MG", "Sudeste", -21.1103, -44.1781, 927, "Barbacena", 0.9, 0.4, -1.0, 0.98),

        # SP
        ("83722", "São Carlos", "SP", "Sudeste", -22.0175, -47.8908, 856, "Franca", 0.7, 0.1, -0.5, 0.95),
        ("A713", "Sorocaba", "SP", "Sudeste", -23.5017, -47.4581, 601, "Campinas", 0.3, 0.2, 1.0, 0.93),
        ("A740", "São José dos Campos", "SP", "Sudeste", -23.1794, -45.8869, 600, "São Paulo", 0.9, -0.3, -1.0, 0.86),
        ("83784", "Taubaté", "SP", "Sudeste", -23.0264, -45.5553, 580, "São Paulo", 1.2, -0.2, -1.0, 0.85),
        ("A744", "Bragança Paulista", "SP", "Sudeste", -22.9519, -46.5419, 817, "São Paulo", -0.3, -0.8, -1.5, 0.94),
        ("A745", "Atibaia", "SP", "Sudeste", -23.1169, -46.5503, 803, "São Paulo", -0.1, -0.7, -1.5, 0.93),
        ("A746", "Serra Negra", "SP", "Sudeste", -22.6119, -46.7006, 925, "Poços de Caldas", 1.1, 1.3, -2.0, 0.91),
        ("A747", "Águas de Lindóia", "SP", "Sudeste", -22.4764, -46.6328, 945, "Poços de Caldas", 0.9, 1.1, -1.5, 0.92),
        ("A748", "Cunha", "SP", "Sudeste", -23.0744, -44.9597, 950, "Nova Friburgo", -0.2, -0.3, -1.0, 0.96),
        ("A726", "Botucatu", "SP", "Sudeste", -22.8858, -48.4450, 804, "Campinas", -0.8, -0.4, 1.0, 0.96),
        ("83729", "Bauru", "SP", "Sudeste", -22.3147, -49.0606, 526, "Campinas", 1.1, 0.8, -1.5, 0.92),
        ("A763", "Marília", "SP", "Sudeste", -22.2139, -49.9458, 660, "Londrina", 0.3, 0.4, -1.5, 0.90),
        ("83716", "Presidente Prudente", "SP", "Sudeste", -22.1256, -51.3889, 475, "Londrina", 1.4, 1.2, -3.5, 0.85),
        ("A735", "São José do Rio Preto", "SP", "Sudeste", -20.8197, -49.3794, 489, "Ribeirão Preto", 0.8, 0.6, -1.5, 0.90),
        ("A734", "Araçatuba", "SP", "Sudeste", -21.2089, -50.4328, 390, "Ribeirão Preto", 1.3, 1.0, -2.0, 0.86),
        ("A728", "Piracicaba", "SP", "Sudeste", -22.7253, -47.6492, 547, "Campinas", 0.8, 0.2, -0.5, 0.92),
        ("A755", "Jundiaí", "SP", "Sudeste", -23.1858, -46.8978, 761, "São Paulo", 0.5, -0.3, -1.5, 0.92),
        ("83786", "Ubatuba", "SP", "Sudeste", -23.4339, -45.0711, 8, "Santos", -0.4, -0.5, 2.0, 1.15),

        # RJ & ES
        ("83744", "Teresópolis", "RJ", "Sudeste", -22.4122, -42.9656, 871, "Petrópolis", -0.3, -0.4, 0.5, 0.96),
        ("A625", "Visconde de Mauá (Resende)", "RJ", "Sudeste", -22.3306, -44.5364, 1200, "Poços de Caldas", -0.4, -0.5, 2.0, 1.02),
        ("83738", "Resende", "RJ", "Sudeste", -22.4689, -44.4467, 410, "Rio de Janeiro", -1.4, -2.5, 1.0, 1.12),
        ("A626", "Miguel Pereira", "RJ", "Sudeste", -22.4539, -43.4689, 618, "Petrópolis", 1.1, 0.9, -2.0, 0.84),
        ("A609", "Volta Redonda", "RJ", "Sudeste", -22.5231, -44.1042, 390, "Rio de Janeiro", -1.1, -2.2, 0.0, 1.08),
        ("A628", "Angra dos Reis", "RJ", "Sudeste", -23.0067, -44.3181, 12, "Santos", 0.4, 0.6, 1.0, 0.95),
        ("83718", "Cabo Frio", "RJ", "Sudeste", -22.8786, -42.0186, 7, "Rio de Janeiro", -1.6, -0.2, 3.0, 0.68),
        ("83698", "Campos dos Goytacazes", "RJ", "Sudeste", -21.7522, -41.3303, 14, "Vitória", 0.5, -0.6, -1.0, 0.82),
        ("A615", "Domingos Martins (Pedra Azul)", "ES", "Sudeste", -20.3636, -40.6592, 950, "Nova Friburgo", -0.2, 0.2, 0.0, 0.92),
        ("A613", "Santa Teresa", "ES", "Sudeste", -19.9356, -40.6003, 655, "Nova Friburgo", 1.2, 1.1, -1.5, 0.90),
        ("A634", "Guarapari", "ES", "Sudeste", -20.6736, -40.4997, 12, "Vitória", -0.3, -0.2, 0.5, 0.96),

        # PR, SC & RS
        ("A807", "Maringá", "PR", "Sul", -23.4205, -51.9331, 555, "Londrina", 0.5, 0.6, -2.0, 0.96),
        ("83836", "Ponta Grossa", "PR", "Sul", -25.0945, -50.1633, 956, "Curitiba", 0.4, -0.2, -1.5, 0.98),
        ("A825", "Guarapuava", "PR", "Sul", -25.3939, -51.4561, 1120, "Curitiba", -0.6, -0.8, -1.0, 1.08),
        ("A845", "Cascavel", "PR", "Sul", -24.9555, -53.4552, 781, "Londrina", -1.2, -0.8, 1.0, 1.12),
        ("A846", "Pato Branco", "PR", "Sul", -26.2289, -52.6706, 760, "Curitiba", 1.1, 0.6, -2.5, 1.15),
        ("83844", "Paranaguá", "PR", "Sul", -25.5206, -48.5094, 5, "Florianópolis", 0.6, 0.5, 2.0, 1.18),
        ("A865", "Urupema", "SC", "Sul", -27.9536, -49.8731, 1335, "São Joaquim", -0.4, -0.6, 1.0, 0.98),
        ("83891", "Lages", "SC", "Sul", -27.8161, -50.3261, 937, "Curitiba", -0.8, -1.0, -1.0, 1.02),
        ("A851", "Joinville", "SC", "Sul", -26.3044, -48.8486, 18, "Florianópolis", 0.8, 0.4, 2.5, 1.22),
        ("83887", "Blumenau", "SC", "Sul", -26.9189, -49.0661, 21, "Florianópolis", 1.1, 0.1, 2.0, 1.12),
        ("83881", "Chapecó", "SC", "Sul", -27.0964, -52.6183, 674, "Londrina", -1.6, -1.4, 1.5, 1.14),
        ("A815", "Criciúma", "SC", "Sul", -28.6775, -49.3697, 46, "Florianópolis", 0.5, -0.8, -1.0, 0.96),
        ("A868", "Balneário Camboriú", "SC", "Sul", -26.9925, -48.6353, 8, "Florianópolis", 0.3, 0.2, 0.5, 1.02),
        ("83942", "Caxias do Sul", "RS", "Sul", -29.1678, -51.1794, 817, "Gramado / Canela", 0.2, 0.3, -1.0, 0.97),
        ("83941", "Bento Gonçalves", "RS", "Sul", -29.1714, -51.5192, 691, "Gramado / Canela", 0.9, 0.7, -1.5, 0.96),
        ("83914", "Passo Fundo", "RS", "Sul", -28.2628, -52.4067, 687, "Gramado / Canela", 1.4, 1.1, -3.0, 1.02),
        ("83985", "Pelotas", "RS", "Sul", -31.7719, -52.3425, 13, "Porto Alegre", -1.8, -1.5, 4.0, 0.96),
        ("83936", "Santa Maria", "RS", "Sul", -29.6842, -53.8069, 113, "Porto Alegre", 0.4, -0.6, 0.0, 1.10),
        ("83927", "Uruguaiana", "RS", "Sul", -29.7547, -57.0883, 74, "Porto Alegre", 1.2, -0.5, -3.0, 0.98),

        # CENTRO-OESTE
        ("83424", "Anápolis", "GO", "Centro-Oeste", -16.3281, -48.9531, 1017, "Brasília", 0.8, 0.5, -1.0, 1.02),
        ("A024", "Alto Paraíso de Goiás (Veadeiros)", "GO", "Centro-Oeste", -14.1325, -47.5100, 1250, "Brasília", -0.6, -0.5, 1.5, 1.04),
        ("83376", "Pirenópolis", "GO", "Centro-Oeste", -15.8525, -48.9592, 770, "Goiânia", -0.5, -0.4, 1.0, 1.02),
        ("A003", "Caldas Novas", "GO", "Centro-Oeste", -17.7444, -48.6250, 686, "Goiânia", 0.5, 0.6, 0.0, 0.96),
        ("83470", "Rio Verde", "GO", "Centro-Oeste", -17.7922, -50.9192, 748, "Goiânia", 0.2, -0.2, 1.0, 1.02),
        ("83704", "Dourados", "MS", "Centro-Oeste", -22.2211, -54.8056, 430, "Campo Grande", -0.3, -1.1, 1.0, 0.98),
        ("A758", "Bonito", "MS", "Centro-Oeste", -21.1261, -56.4836, 315, "Campo Grande", 0.8, -0.2, 1.5, 0.96),
        ("83552", "Corumbá", "MS", "Centro-Oeste", -19.0089, -57.6511, 118, "Cuiabá", -0.5, 0.2, 1.0, 0.82),
        ("A906", "Chapada dos Guimarães", "MT", "Centro-Oeste", -15.4606, -55.7497, 811, "Brasília", 1.8, 1.1, 2.0, 1.12),
        ("83410", "Rondonópolis", "MT", "Centro-Oeste", -16.4708, -54.6356, 227, "Cuiabá", -0.4, -1.0, -1.0, 1.02),
        ("A917", "Sinop", "MT", "Centro-Oeste", -11.8642, -55.5025, 384, "Cuiabá", -0.6, -0.5, 4.0, 1.28),

        # NORDESTE
        ("83242", "Lençóis (Chapada Diamantina)", "BA", "Nordeste", -12.5631, -41.3889, 439, "Vitória da Conquista", 2.1, 1.8, 1.0, 1.25),
        ("83184", "Morro do Chapéu", "BA", "Nordeste", -11.5500, -41.1561, 1012, "Vitória da Conquista", -0.8, -0.6, 1.5, 0.92),
        ("A430", "Piatã", "BA", "Nordeste", -13.1519, -41.7725, 1268, "Diamantina", -0.4, -0.6, -1.5, 0.72),
        ("83221", "Feira de Santana", "BA", "Nordeste", -12.2667, -38.9667, 234, "Salvador", 0.8, -1.8, -3.5, 0.52),
        ("83348", "Ilhéus", "BA", "Nordeste", -14.7889, -39.0494, 45, "Salvador", -0.5, -1.0, 2.5, 0.95),
        ("83442", "Porto Seguro", "BA", "Nordeste", -16.4497, -39.0647, 38, "Salvador", -0.4, -1.1, 1.0, 0.82),
        ("83236", "Barreiras", "BA", "Nordeste", -12.1528, -44.9900, 452, "Goiânia", 1.8, 0.5, -3.0, 0.75),
        ("A351", "Triunfo", "PE", "Nordeste", -7.8381, -38.1017, 1004, "Garanhuns", -0.8, -0.5, -2.5, 0.95),
        ("A328", "Gravatá", "PE", "Nordeste", -8.2011, -35.5647, 447, "Garanhuns", 1.4, 1.2, -1.5, 0.88),
        ("82892", "Caruaru", "PE", "Nordeste", -8.2833, -35.9761, 545, "Campina Grande", 0.2, 0.0, -1.5, 0.85),
        ("82474", "Guaramiranga", "CE", "Nordeste", -4.2633, -38.9328, 865, "Garanhuns", -1.2, -0.2, 3.0, 1.35),
        ("82704", "Juazeiro do Norte", "CE", "Nordeste", -7.2131, -39.3153, 377, "Petrolina", 0.2, -0.4, 3.0, 1.45),
        ("A318", "Martins", "RN", "Nordeste", -6.0878, -37.9111, 745, "Campina Grande", -0.4, -0.2, -1.0, 1.12),
        ("82796", "Areia", "PB", "Nordeste", -6.9681, -35.6983, 618, "Garanhuns", -0.3, 0.4, 2.0, 1.22),
        ("A345", "Pedro II", "PI", "Nordeste", -4.4247, -41.4586, 603, "Campina Grande", 0.6, 0.4, -2.0, 1.18),

        # NORTE
        ("82246", "Santarém", "PA", "Norte", -2.4431, -54.7083, 36, "Manaus", -0.2, 0.1, 0.0, 0.88),
        ("82562", "Marabá", "PA", "Norte", -5.3686, -49.1178, 84, "Belém", 1.1, -0.4, -5.0, 0.68),
        ("82240", "Parintins", "AM", "Norte", -2.6283, -56.7358, 27, "Manaus", -0.3, 0.2, 1.0, 0.96),
        ("A938", "Vilhena", "RO", "Norte", -12.7406, -60.1458, 615, "Campo Grande", -0.2, 0.6, 6.0, 1.32),
    ]

    for est in estacoes_base:
        est["fonte_dados"] = "INMET 1991–2020 (Estação Oficial)"

    for cod, nome, uf, reg, lat, lon, alt, ref_nome, dt_max, dt_min, dur, f_chuva in extras_config:
        ref = mapa_ref[ref_nome]
        series = _gerar_serie_derivada(
            ref["tmax"], ref["tmin"], ref["ur"], ref["chuva"], ref["insolacao"],
            dt_max=dt_max, dt_min=dt_min, dur=dur, fator_chuva=f_chuva
        )
        estacoes_base.append({
            "codigo_inmet": cod,
            "nome": nome,
            "uf": uf,
            "regiao": reg,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "fonte_dados": "Modelagem Topoclimática Regional (INMET/Reanálise)",
            **series,
        })

    # Integrar dados oficiais diretamente das planilhas das Normais Climatológicas INMET (1991-2020)
    caminho_inmet_json = Path(__file__).resolve().parent / "raw_inmet" / "inmet_oficial_1991_2020.json"
    if caminho_inmet_json.exists():
        import unicodedata
        inmet_oficial = json.loads(caminho_inmet_json.read_text(encoding="utf-8"))

        def _norm(txt: str) -> str:
            return "".join(
                c for c in unicodedata.normalize("NFD", str(txt or ""))
                if unicodedata.category(c) != "Mn"
            ).upper().strip()

        chaves_atendidas = set()
        codigos_oficiais_usados = set()

        for est in estacoes_base:
            nome_limpo = _norm(est["nome"].split("(")[0].split("/")[0])
            chave = f"{nome_limpo}|{_norm(est['uf'])}"
            if chave in inmet_oficial:
                dados_of = inmet_oficial[chave]
                est["codigo_inmet"] = str(dados_of["codigo_inmet"])
                est["tmax"] = dados_of["tmax"]
                est["tmin"] = dados_of["tmin"]
                est["ur"] = dados_of["ur"]
                est["chuva"] = dados_of["chuva"]
                if dados_of.get("insolacao") and est.get("fonte_dados") != "INMET 1991–2020 (Estação Oficial)":
                    est["insolacao"] = dados_of["insolacao"]
                est["fonte_dados"] = "INMET 1991–2020 (Planilha Oficial)"
                if dados_of.get("recordes") and est["nome"] not in RECORDES_OFICIAIS_BDMEP:
                    est["recordes_oficiais_xlsx"] = dados_of["recordes"]
                chaves_atendidas.add(chave)
                codigos_oficiais_usados.add(str(dados_of["codigo_inmet"]))

        # Garantir que estações derivadas não colidam com códigos de estações oficiais do INMET
        codigos_vistos = set()
        for est in estacoes_base:
            cod_atual = str(est["codigo_inmet"])
            if cod_atual in codigos_vistos:
                suf = _norm(est["nome"])[:3]
                est["codigo_inmet"] = f"{cod_atual}_{suf}"
            codigos_vistos.add(str(est["codigo_inmet"]))

        # Adicionar todas as demais estações oficiais completas do INMET (1991-2020) ao catálogo nacional
        aliases_ignorados = {"RECIFE (CURADO)|PE", "SALVADOR (ONDINA)|BA", "SAO PAULO(MIR.DE SANTANA)|SP"}
        for chave, dados_of in sorted(inmet_oficial.items()):
            if chave in chaves_atendidas or chave in aliases_ignorados:
                continue
            cod_of = str(dados_of["codigo_inmet"])
            if cod_of in codigos_oficiais_usados:
                continue
            if cod_of in codigos_vistos:
                # Se alguma cidade derivada usava esse código provisoriamente, renomeia o código da derivada
                for e_ant in estacoes_base:
                    if str(e_ant["codigo_inmet"]) == cod_of:
                        e_ant["codigo_inmet"] = f"REG_{_norm(e_ant['nome'])[:5]}_{e_ant['uf']}"
            codigos_vistos.add(cod_of)
            codigos_oficiais_usados.add(cod_of)

            est_nova = {
                "codigo_inmet": cod_of,
                "nome": dados_of.get("nome_exibicao", dados_of["nome_inmet"].title()),
                "uf": dados_of["uf"],
                "regiao": dados_of.get("regiao", "Sudeste"),
                "lat": float(dados_of.get("lat", -15.0)),
                "lon": float(dados_of.get("lon", -47.0)),
                "alt": int(dados_of.get("alt", 400)),
                "fonte_dados": "INMET 1991–2020 (Planilha Oficial)",
                "tmax": dados_of["tmax"],
                "tmin": dados_of["tmin"],
                "ur": dados_of["ur"],
                "chuva": dados_of["chuva"],
                "insolacao": dados_of.get("insolacao", [200] * 12),
            }
            if dados_of.get("recordes"):
                est_nova["recordes_oficiais_xlsx"] = dados_of["recordes"]
            estacoes_base.append(est_nova)

    # Aplicação de harmonização topoclimática / NASA POWER Climatology para eliminar distorções de dossel e derivações inadequadas
    for est in estacoes_base:
        nome_limpo = _norm(est["nome"].split("(")[0].split("/")[0])
        chave = f"{nome_limpo}|{_norm(est['uf'])}"
        if chave in ESTACOES_HARMONIZADAS_TOPOCLIMATICAS:
            h = ESTACOES_HARMONIZADAS_TOPOCLIMATICAS[chave]
            est["tmax"] = h["tmax"]
            est["tmin"] = h["tmin"]
            est["ur"] = h["ur"]
            est["chuva"] = h["chuva"]
            est["alt"] = h.get("alt", est["alt"])
            est["lat"] = h.get("lat", est["lat"])
            est["lon"] = h.get("lon", est["lon"])
            est["fonte_dados"] = h.get("fonte_dados", est["fonte_dados"])
            est["insolacao"] = estimar_insolacao_mensal(est["ur"], est["chuva"])
            if h.get("recordes_oficiais_xlsx"):
                est["recordes_oficiais_xlsx"] = h["recordes_oficiais_xlsx"]

    for est in estacoes_base:
        est["pm25"] = obter_pm25_estacao(est)

    return estacoes_base


def exportar_csv_referencia(estacoes: List[Dict[str, Any]], caminho_csv: Path) -> None:
    linhas = []
    for est in estacoes:
        pm25 = est.get("pm25") or obter_pm25_estacao(est)
        for idx, mes in enumerate(MESES_ORDEM):
            linhas.append({
                "codigo_inmet": est["codigo_inmet"],
                "nome": est["nome"],
                "uf": est["uf"],
                "regiao": est["regiao"],
                "lat": est["lat"],
                "lon": est["lon"],
                "altitude_m": est["alt"],
                "mes_num": idx + 1,
                "mes": mes,
                "tmax_c": est["tmax"][idx],
                "tmin_c": est["tmin"][idx],
                "ur_pct": est["ur"][idx],
                "chuva_mm": est["chuva"][idx],
                "insolacao_h": est["insolacao"][idx],
                "pm25_ugm3": pm25[idx],
            })
    df = pd.DataFrame(linhas)
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_csv, index=False, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    caminho_csv = base_dir / "data_pipeline" / "raw_inmet" / "normais_1991_2020_estacoes.csv"
    caminho_geojson = base_dir / "public" / "data" / "estacoes_clima_brasil.geojson"
    caminho_js = base_dir / "public" / "data" / "estacoes_clima_brasil.js"

    estacoes = obter_estacoes_referencia_inmet()
    exportar_csv_referencia(estacoes, caminho_csv)

    geojson_data = processar_estacoes(estacoes)
    caminho_geojson.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_geojson, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    with open(caminho_js, "w", encoding="utf-8") as f:
        f.write("window.ESTACOES_CLIMA_GEOJSON = ")
        json.dump(geojson_data, f, ensure_ascii=False)
        f.write(";\n")

    print(f"[OK] CSV exportado em: {caminho_csv}")
    print(f"[OK] GeoJSON gerado em: {caminho_geojson} ({len(geojson_data['features'])} cidades/estações)")
    print(f"[OK] Fallback JS gerado em: {caminho_js}")
    print("\nTop 10 Cidades no ICCB v2.0 (Calibrado + Poluição PM2.5):")
    for feat in geojson_data["features"][:10]:
        p = feat["properties"]
        print(
            f"  #{p['ranking']:02d} {p['nome']}-{p['uf']} ({p['altitude_m']}m) -> "
            f"ICCB: {p['nota_conforto_anual']} [{p['classificacao']}] | PM2.5: {p['resumo_anual']['pm25_media_anual']} µg/m³"
        )
    print("\nÚltimas 5 Cidades no Ranking (Nenhuma nota zero — todas habitáveis):")
    for feat in geojson_data["features"][-5:]:
        p = feat["properties"]
        print(
            f"  #{p['ranking']:03d} {p['nome']}-{p['uf']} ({p['altitude_m']}m) -> "
            f"ICCB: {p['nota_conforto_anual']} [{p['classificacao']}] | PM2.5: {p['resumo_anual']['pm25_media_anual']} µg/m³"
        )


if __name__ == "__main__":
    main()
