# Test contracts

Three pairs of scanned-style JPGs simulating real legal documents in Spanish. Each pair consists of an **original contract** and its **amendment / adenda**, drawn from the example corpus that ships with the bootcamp's Module 4 final project.

The expected changes below are sourced from the bootcamp's ground-truth document (`Diferencias entre Contratos y Enmiendas de ejemplo.docx`) and are reproduced here so the demo can be cross-checked without leaving the repo.

## Pair 1 — Software License (`contract_1_*`)

- **Files:** `contract_1_original.jpg`, `contract_1_amendment.jpg`
- **Parties:** TechNova S.A. (Licenciante) and DataBridge Soluciones S.R.L. (Licenciatario)
- **Type:** Software license for "NovaAnalytics"
- **Expected changes (5, includes a brand-new clause):**

| Section | Original | Amendment |
| --- | --- | --- |
| 2. Plazo | 12 meses | **24 meses** |
| 3. Pago | USD 12.000 | **USD 15.000** |
| 4. Soporte | email | **email + chat** |
| 5. Terminación | 30 días | **60 días** |
| 7. Protección de Datos | — | **Nueva cláusula** |

## Pair 2 — Consulting Services (`contract_2_*`)

- **Files:** `contract_2_original.jpg`, `contract_2_amendment.jpg`
- **Parties:** Orion Consulting Group (Consultor) and GreenWave Energía S.A. (Cliente)
- **Type:** Strategic consulting services for renewable energy
- **Expected changes (5, includes a brand-new clause):**

| Section | Original | Amendment |
| --- | --- | --- |
| 1. Alcance del Servicio | Consultoría estratégica | **+ análisis regulatorio** |
| 2. Duración | 6 meses | **9 meses** |
| 3. Honorarios | USD 8.000 mensuales | **USD 9.500 mensuales** |
| 4. Entregables | reportes mensuales | **reportes quincenales** |
| 7. Propiedad Intelectual | — | **Nueva cláusula** |

## Pair 3 — SaaS Agreement (`contract_3_*`)

- **Files:** `contract_3_original.jpg`, `contract_3_amendment.jpg`
- **Parties:** CloudMetrics Ltd. (Proveedor) and RetailPulse S.A. (Cliente)
- **Type:** SaaS subscription for the CloudMetrics analytics platform
- **Expected changes (3, no new clauses — simpler diff):**

| Section | Original | Amendment |
| --- | --- | --- |
| 3. Precio | USD 1.200 | **USD 1.250** |
| 4. Disponibilidad del Servicio | 99,5% | **99,9%** |
| 5. Soporte | email | **email + sistema de tickets** |

## Live-demo recommendation

For the 30-minute defense, run **Pair 1** (complex — 5 changes + new clause) and **Pair 3** (simple — 3 changes, no new clauses) to exercise both ends of the difficulty spectrum.
