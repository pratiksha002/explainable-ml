def compare_explanations(shap_values, lime_values):
    agreement = []
    differences = []

    lime_features = {}
    for rule, value in lime_values.items():
        feature = rule.split()[0]
        lime_features[feature] = value

    for feature, shap_val in shap_values.items():
        lime_val = lime_features.get(feature)

        if lime_val is None:
            continue


        if (shap_val > 0 and lime_val > 0) or (shap_val < 0 and lime_val < 0):
            agreement.append(f"{feature} has consistent impact in SHAP and LIME")

        else:
            differences.append(f"{feature} impact differs between SHAP and LIME")


        if abs(shap_val - lime_val) > 0.5:
            differences.append(f"{feature} has significantly different contribution values")

        return {
            "agreement": agreement,
            "differnces": differences
        }
