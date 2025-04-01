import google.generativeai as genai
import pickle
import numpy as np
import streamlit as st


st.success("just the first page")
# Load ML models
model_paths = {
    "knn": "C:\\Users\\Bhavya\\Downloads\\drive-download-20250331T152810Z-001\\knn_model.sav",
    "nbayes": "C:\\Users\\Bhavya\\Downloads\\drive-download-20250331T152810Z-001\\NBayes.sav",
    "rforest": "C:\\Users\\Bhavya\\Downloads\\drive-download-20250331T152810Z-001\\Rforest.sav",
    "svm": "C:\\Users\\Bhavya\\Downloads\\drive-download-20250331T152810Z-001\\svm_model_sav.sav",
    "Dtree":"C:\\Users\\Bhavya\\Downloads\\drive-download-20250331T152810Z-001\\DTree.sav"
}
models = {}
for key, path in model_paths.items():
    with open(path, 'rb') as file:
        models[key] = pickle.load(file)

symptoms_list = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine',
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum',
    'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
    'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

while len(symptoms_list) < models['knn'].n_features_in_:
    symptoms_list.append(f"dummy_feature_{len(symptoms_list)}")


detected_symptoms_vector = []


def extract_symptoms(user_input):
    global detected_symptoms_vector
    detected_symptoms = [symptom for symptom in symptoms_list if symptom in user_input.lower()]
    detected_symptoms_vector.extend(detected_symptoms)
    return detected_symptoms


def predict_condition(symptoms):
    input_vector = np.zeros(len(symptoms_list))
    for symptom in symptoms:
        if symptom in symptoms_list:
            input_vector[symptoms_list.index(symptom)] = 1
    input_vector = input_vector.reshape(1, -1)

    predicted_conditions = set()
    for model in models:
        condition = models[model].predict(input_vector)[0]
        predicted_conditions.add(condition)
    return list(predicted_conditions)


# Set up your Google API key
genai.configure(api_key="AIzaSyCFRRix7b4V_0HMZ3o9jYgxQLsAcx1C2yc")


def valid_response(user_input):
    symptoms = extract_symptoms(user_input)
    if symptoms:
        print(f"Detected symptoms: {', '.join(symptoms)}. Do you have any more symptoms? (yes/no)")
        while True:
            more_input = input("You: ").lower()
            if more_input == "no":
                predictions = predict_condition(detected_symptoms_vector)
                print("Predicted conditions:", predictions)
                summary = genai.GenerativeModel("gemini-1.5-pro-002").generate_content(f"Provide a 50 words summary for each of the medical conditions mentioned here along with thier treatment and symptoms {', '.join(predictions)}.")
                print("Advice:", summary.text)
                break
            elif more_input == "yes":
                additional_input = input("Describe more symptoms: ")
                extract_symptoms(additional_input)
            else:
                print("Please answer with 'yes' or 'no'.")
    else:
        # Pass input to Gemini if no symptoms are detected
        try:
            response_pro = genai.GenerativeModel("gemini-1.5-pro-002").generate_content(user_input)
            print("Chatbot:", response_pro.text)
        except Exception as e:
            print("Error:", e)


def medical_chatbot():
    print("Medical Chatbot: Ask me anything about health and medicine. Type 'exit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Take care! Goodbye.")
            break
        valid_response(user_input)


if __name__ == "__main__":
    medical_chatbot()
