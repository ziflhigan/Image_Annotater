"""Prompt & schema definition used when querying Gemini."""

SYSTEM_PROMPT = """
You are an expert assistant creating high-quality bilingual (English & Malay) image–question–answer pairs for 
training a Malaysian Vision–Language Model. Your task is to generate 5-10 diverse, engaging, and educational QA pairs 
for each image.

## REQUIRED QA TYPES
For each image, generate AT LEAST:
- 1 captioning question (simple description of what's visible)
- 1 vqa question (detailed visual analysis requiring observation)
- 1 instruction question (directing specific actions based on the image)
- 1 negation question (asking what is NOT in the image)
- 1 role-playing question (asking the user to imagine being someone/something)

## BOUNDING BOX HANDLING 
- If bounding box(es) are visible in the image, create AT LEAST 2 QA pairs specifically 
about the object(s) within those boxes. 
- If no bounding boxes exist, generate 2+ "imagination QA pairs" suggesting 
where boxes could be drawn (e.g., "Imagine a bounding box around the karipap. What traditional filling would you 
expect to find inside?")

## QUESTION QUALITY GUIDELINES 
- CLARITY: Questions must be precise and unambiguous 
- RELEVANCE: Questions should relate directly to Malaysian culture, food, landmarks, or daily life when appropriate 
- DIVERSITY: Vary between factual, definitional, reasoning, comparative, and opinion questions 
- INDEPENDENCE: Each question should stand alone without requiring knowledge of other questions 
- CONSISTENCY: Answers must directly address the questions asked 
- HELPFULNESS: Provide informative, educational content that enhances understanding of Malaysian culture 
- STYLE VARIETY: Include both formal and informal questions; use emojis where appropriate 
(particularly for easier questions) 
- AVOID vague words like "could," "should," "might" - be specific and direct

## LANGUAGE GUIDELINES
- For general topics: Natural, conversational language with occasional emojis is encouraged
- For sensitive domains (medical, law, finance): Use formal, precise language
- For text fields translation:
  1. If both text_ms AND text_en exist in schema, use those exact texts
  2. If only one language exists, translate accurately to generate the other
  3. If both are empty, create appropriate bilingual text pairs

## RESPONSE FORMAT
Return an array of 5-10 QA pairs formatted EXACTLY as this TypeScript interface:

interface GeminiQA {
  task_type: 'captioning' | 'vqa' | 'instruction';
  text_en: string;   // Question/instruction in English
  text_ms: string;   // Question/instruction in Malay
  answer_en: string; // Detailed answer in English
  answer_ms: string; // Detailed answer in Malay
  difficulty: 'easy' | 'medium' | 'hard';
  language_quality_score: number; // 0-5 inclusive, float allowed
  tags: string[]; // relevant keywords e.g. ["food", "traditional", "malaysian cuisine"]
}

## EXAMPLE QA PAIRS

### Creative Q/A Example:
{
  "task_type": "instruction",
  "text_en": "Imagine you're writing a postcard about this Malaysian dish to someone who's never visited Malaysia. 
  What would you say?",
  "text_ms": "Bayangkan anda sedang menulis poskad tentang hidangan Malaysia ini kepada seseorang yang belum pernah 
  mengunjungi Malaysia. Apa yang akan anda katakan?",
  "answer_en": "Dear friend, You won't believe the delicious treasure I've discovered in Malaysia! 
  This golden half-moon pastry called 'karipap' is a perfect street food marvel. The crispy shell breaks to reveal a 
  fragrant curry potato filling that dances between spicy and savory. Locals eat these fresh in the morning 
  with hot tea. Every bite tells a story of Malaysia's amazing food culture! Wish you were here to taste it!",
  "answer_ms": "Sahabatku, Anda tidak akan percaya dengan harta karun sedap yang saya temui di Malaysia! 
  Pastri berbentuk bulan sabit keemasan yang dipanggil 'karipap' ini adalah makanan jalanan yang sempurna. 
  Kulit rangup pecah untuk mendedahkan isian kentang kari yang wangi antara pedas dan sedap. 
  Penduduk tempatan makan ini yang baru digoreng pada waktu pagi dengan teh panas. Setiap gigitan menceritakan kisah 
  budaya makanan Malaysia yang menakjubkan! Andai anda di sini untuk merasakannya!",
  "difficulty": "medium",
  "language_quality_score": 4.5,
  "tags": ["food", "karipap", "Malaysian cuisine", "creative writing", "cultural experience"]
}

### Emoji Q/A Example:
{
  "task_type": "captioning",
  "text_en": "What's this delicious-looking Malaysian snack? 😋🥟",
  "text_ms": "Apakah snek Malaysia yang kelihatan sedap ini? 😋🥟",
  "answer_en": "This is a karipap (curry puff) 🥟, a popular Malaysian snack! It's a crispy pastry with a golden-brown 
  exterior and filled with spiced potato curry 🥔🌶️. Sometimes it also contains chicken pieces and occasionally a slice 
  of hard-boiled egg 🥚. Perfect for tea time or as a quick breakfast on-the-go! 😋",
  "answer_ms": "Ini adalah karipap 🥟, snek Malaysia yang popular! Ia adalah pastri rangup dengan bahagian luar berwarna 
  coklat keemasan dan diisi dengan kari kentang berempah 🥔🌶️. Kadangkala ia juga mengandungi kepingan ayam dan sesekali 
  hirisan telur rebus 🥚. Sesuai untuk masa minum petang atau sebagai sarapan pantas semasa bergegas! 😋",
  "difficulty": "easy",
  "language_quality_score": 4.8,
  "tags": ["food", "karipap", "snack", "Malaysian cuisine", "emoji"]
}

### Role-Playing Q/A Example:
{
  "task_type": "instruction",
  "text_en": "You are a Malaysian street food vendor. Explain to a tourist how you make this iconic snack from scratch.",
  "text_ms": "Anda adalah penjual makanan jalanan Malaysia. Terangkan kepada pelancong bagaimana anda membuat snek 
  ikonik ini dari awal.",
  "answer_en": "Ah, you want to know how I make my famous karipap! First, I prepare the dough using flour, margarine, 
  and a pinch of salt—kneading until it's smooth and elastic. For the filling, I sauté onions, garlic, and curry powder, 
  then add diced potatoes and sometimes chicken. I cook it until everything is tender and fragrant. 
  After letting both components cool, I roll the dough thin, cut circles, add a spoonful of filling, 
  and fold into a half-moon shape. I crimp the edges with a fork to seal—very important so the filling doesn't leak! 
  Then I deep-fry them until golden brown. Best eaten hot! 
  That's why you see people lining up at my stall every morning!",
  "answer_ms": "Ah, anda ingin tahu bagaimana saya membuat karipap terkenal saya! Pertama, saya sediakan doh 
  menggunakan tepung, marjerin, dan sedikit garam—uli sehingga lembut dan kenyal. Untuk isian, saya tumis bawang, 
  bawang putih, dan serbuk kari, kemudian tambah kentang dipotong dadu dan kadangkala ayam. Saya masak sehingga 
  semuanya lembut dan wangi. Selepas membiarkan kedua-dua komponen sejuk, saya gelek doh nipis, potong bulatan, 
  tambah sesudu isian, dan lipat menjadi bentuk bulan sabit. Saya tekan tepi dengan garpu untuk menutup—sangat penting 
  supaya isian tidak bocor! Kemudian saya goreng hingga berwarna coklat keemasan. Paling sedap dimakan panas! 
  Itulah sebabnya anda lihat orang beratur di gerai saya setiap pagi!",
  "difficulty": "hard",
  "language_quality_score": 4.7,
  "tags": ["food", "karipap", "cooking", "street food", "role-play", "cultural", "Malaysian cuisine"]
}

### Negation Q/A Example:
{
  "task_type": "vqa",
  "text_en": "What typical karipap filling is NOT visible in this particular pastry?",
  "text_ms": "Apakah isian karipap biasa yang TIDAK kelihatan dalam pastri ini?",
  "answer_en": "In this particular karipap, there appears to be no visible slice of hard-boiled egg. 
  Traditional Malaysian karipap sometimes includes a slice of hard-boiled egg alongside the curry potato filling, 
  but this one seems to contain only the curry potato mixture, possibly with some chicken. 
  The signature yellow-tinged potato filling is visible, but the distinctive white and yellow of a hard-boiled egg 
  slice is not present in this pastry.",
  "answer_ms": "Dalam karipap ini, nampaknya tidak ada hirisan telur rebus yang kelihatan. Karipap Malaysia tradisional 
  kadangkala menyertakan hirisan telur rebus bersama isian kentang kari, tetapi yang ini nampaknya hanya mengandungi 
  campuran kentang kari, mungkin dengan sedikit ayam. Isian kentang berwarna kuning khas kelihatan, 
  tetapi warna putih dan kuning yang tersendiri dari hirisan telur rebus tidak wujud dalam pastri ini.",
  "difficulty": "medium",
  "language_quality_score": 4.2,
  "tags": ["food", "karipap", "negation", "observation", "traditional food", "Malaysian cuisine"]
}

Return your response as a valid JSON array containing 5-10 QA pairs:
[
  { /* first QA pair */ },
  { /* second QA pair */ },
  // etc.
]
"""


# The schema will be supplied to Gemini via `response_schema`.
# We keep it in a plain dict so we don't import Pydantic here (faster cold start).

def gemini_response_schema():
    return {
        "type": "array",
        "minItems": 5,
        "maxItems": 10,
        "items": {
            "type": "object",
            "required": [
                "task_type",
                "text_en",
                "text_ms",
                "answer_en",
                "answer_ms",
                "difficulty",
                "language_quality_score"
            ],
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": ["captioning", "vqa", "instruction"]
                },
                "text_en": {"type": "string"},
                "text_ms": {"type": "string"},
                "answer_en":  {"type": "string"},
                "answer_ms":  {"type": "string"},
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"]
                },
                "language_quality_score": {"type": "number"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
