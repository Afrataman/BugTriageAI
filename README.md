# BugTriage AI

BugTriage AI is a machine learning application that classifies software issue descriptions into three categories:

- Bug
- Feature
- Documentation

The project uses a custom-trained text classification model instead of relying only on an external AI API.

## Project Purpose

Software teams receive many issue reports, feature requests, and documentation requests. Manually categorizing these records can take time.

BugTriage AI analyzes the entered issue description and predicts its category together with a confidence score.

Example:

```text
Input:
Login butonuna basıldığında uygulama kapanıyor.

Prediction:
Bug

Confidence:
52.36%