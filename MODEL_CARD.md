# Model Card: Bank + Telecom Churn Screening

## Bank
Predictors: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography.

Excluded identifiers: CustomerId, Surname.

Duplicate rows removed: 2. Train/validation/test: 6400/1600/2000.

Model: XGBoost; validation-selected F2 cutoff: 0.425; quality gate: demonstration.


## Telecom
Predictors: SeniorCitizen, tenure, MonthlyCharges, TotalCharges, Partner, Dependents, PhoneService, InternetService, Contract, PaymentMethod.

Excluded identifiers: customerID.

Duplicate rows removed: 0. Train/validation/test: 3763/941/1176.

Model: XGBoost; validation-selected F2 cutoff: 0.100; quality gate: research_only_low_signal.


## Methodology
Training-only preprocessing, validation PR AUC model selection, validation F2 threshold selection, untouched test performance.

## Limitations
Educational datasets, distribution shift, no probability calibration, no subgroup fairness validation, and no downstream business impact study. For demonstration only.
