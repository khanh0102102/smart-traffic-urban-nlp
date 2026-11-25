# ST-NLP project structure

```text
smart-traffic-urban-nlp/
├── backend/
│   └── app/
│       ├── api/routes/
│       ├── core/
│       ├── repositories/
│       ├── schemas/
│       ├── services/
│       └── main.py
├── frontend/
│   ├── citizen/
│   ├── admin/
│   └── assets/{css,js}/
├── ml/
│   ├── data/{raw,interim,processed}/
│   ├── src/{preprocessing,features,models,evaluation}/
│   ├── notebooks/
│   ├── artifacts/
│   └── reports/
├── database/{migrations,seeds}/
├── docs/
├── scripts/
└── tests/{backend,ml,integration}/
```

The backend owns API behavior; the frontend owns presentation; the ML workspace owns model development; the database workspace owns schema evolution. Phase 1 provides only the folders, configuration and minimal health/portal entry points needed to validate the repository.
