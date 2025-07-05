from transformers import pipeline
#step 1 load the model 
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
# step 2 add islamic emotions label
islamic_responses = {
    "sadness": "Aur beshak Sabr karne walon ke saath Allah hota hai. (Surah Baqarah 2:153)",
    "anger": "Jo gussa pee gaya, wo Allah ka mehboob bana. (Hadith)",
    "joy": "Khushi ka shukar ada karo — Allah shukar guzar ko aur deta hai. (Ibrahim 14:7)",
    "love": "Jo Allah se mohabbat karta hai, woh har mohabbat me barkat paata hai.",
    "fear": "Na ghabrao, main tumhare saath hoon. (Surah Taha 20:46)"
}
# step 3 user sy input lo 
user_input = input("Apna jazbaat bayan karein: ")
# step 4 emotion predict karo
result = emotion_model(user_input)
top_emotion = max(result[0], key=lambda x: x['score'])
emotion = top_emotion['label']
confidence = round(top_emotion['score']*100, 2)
response = islamic_responses.get(emotion, "Koi khaas jawab nahi hai.")
print("AI analysis")
print(f"emotion: {emotion} ({confidence}%)")
print(f" islamic response: {response}")

