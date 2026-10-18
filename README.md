# Plataforma de Conforto Climático e Histórico Meteorológico do Brasil (ICCB v1.0)

Aplicação Web Responsiva (Desktop + Mobile/PWA) e Pipeline de Engenharia de Dados para avaliação bioclimática dos municípios brasileiros com base nas **Normais Climatológicas do INMET (1991–2020)** e na reanálise via satélite **NASA POWER API**.

---

## Estrutura do Projeto

```text
site/
├── data_pipeline/
│   ├── raw_inmet/
│   │   └── normais_1991_2020_estacoes.csv   # Planilha consolidada das séries mensais (gerada pelo ETL)
│   ├── processador_clima_inmet.py           # Motor matemático ICCB v1.0 e gerador do GeoJSON
│   ├── buscador_nasa_power.py               # Consulta NASA POWER Climatology para qualquer coordenada
│   └── requirements.txt                     # Dependências Python (pandas, numpy, openpyxl, requests)
├── public/
│   ├── data/
│   │   ├── estacoes_clima_brasil.geojson    # Base geoespacial consumida pelo Front-End
│   │   └── estacoes_clima_brasil.js         # Bundle de fallback para execução direta via file://
│   ├── icons/                               # Ícones vetoriais do PWA (192x192 e 512x512)
│   ├── manifest.json                        # Manifesto de instalação PWA
│   ├── sw.js                                # Service Worker para cache offline
│   └── index.html                           # Aplicação Web Responsiva (Leaflet + Chart.js + Tailwind)
└── README.md
```

---

## Como Executar

### 1. Abrir a Aplicação Web Imediatamente
* **Opção A (Servidor Local com suporte completo a PWA):**
  ```bash
  python -m http.server 8080 --directory public
  ```
  Acesse `http://localhost:8080` no navegador.
* **Opção B (Abertura Direta):**
  Basta abrir o arquivo `public/index.html` diretamente no navegador (possui fallback automático embutido via `public/data/estacoes_clima_brasil.js`).

### 2. Reprocessar o Pipeline de Dados (Python ETL)
```bash
pip install -r data_pipeline/requirements.txt
python data_pipeline/processador_clima_inmet.py
```

### 3. Consultar Qualquer Município na NASA POWER API (CLI ou Web)
* **Pela Interface Web:** Clique no botão **"Consultar 5.570 Municípios (NASA POWER)"** no topo do mapa, digite o nome de qualquer cidade brasileira e adicione-a ao mapa e ao ranking em tempo real.
* **Pelo Terminal (Python):**
  ```bash
  python data_pipeline/buscador_nasa_power.py --nome "Monte Verde" --uf "MG" --lat -22.8628 --lon -46.0364
  ```
