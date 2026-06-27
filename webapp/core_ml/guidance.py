import os
import json
from django.conf import settings
from core_ml.rag import get_rag_engine

class GuidanceGenerator:
    def __init__(self):
        self.engine = get_rag_engine()

    def generate_guidance(self, modality: str, findings_text: str) -> dict:
        """
        Formulate a personalized guidance package using RAG chunks.
        Falls back to rule-based clinical templates when Claude API is not configured.
        """
        # Retrieve medical evidence chunks
        chunks = self.engine.retrieve(findings_text, modality, limit=2)
        
        # Build RAG knowledge block for context
        knowledge_context = "\n\n".join([f"Source: {c['title']}\nContent: {c['content']}" for c in chunks])
        
        # Check for Claude API Key
        api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
        
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                
                system_prompt = (
                    "You are a clinical guidance compiler. You receive clinical findings and medical guidelines.\n"
                    "Your task is to generate a structured JSON object containing personalized patient guidance.\n"
                    "The JSON must have the following structure EXACTLY (do not wrap in markdown or add text, return raw JSON):\n"
                    "{\n"
                    "  \"diet\": {\n"
                    "    \"recommended\": [\"item 1\", \"item 2\"],\n"
                    "    \"avoid\": [\"item 1\", \"item 2\"],\n"
                    "    \"reasoning\": \"nutritional justification\"\n"
                    "  },\n"
                    "  \"exercise\": {\n"
                    "    \"allowed\": [\"activity 1\", \"activity 2\"],\n"
                    "    \"restrictions\": [\"action 1\", \"action 2\"],\n"
                    "    \"guidelines\": \"progressive recovery advice\"\n"
                    "  },\n"
                    "  \"lifestyle\": {\n"
                    "    \"tips\": [\"tip 1\", \"tip 2\"],\n"
                    "    \"red_flags\": [\"red flag 1\", \"red flag 2\"],\n"
                    "    \"notes\": \"general sleep/stress notes\"\n"
                    "  },\n"
                    "  \"routing\": {\n"
                    "    \"specialist\": \"Department Name\",\n"
                    "    \"urgency\": \"Routine/Moderate/High\",\n"
                    "    \"guidance\": \"specialist routing tip\"\n"
                    "  },\n"
                    "  \"questions\": [\n"
                    "    \"Question 1\",\n"
                    "    \"Question 2\"\n"
                    "  ]\n"
                    "}"
                )
                
                user_content = (
                    f"Modality: {modality}\n"
                    f"Findings: {findings_text}\n\n"
                    f"[Retrieved Medical Guidelines]\n{knowledge_context}"
                )
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_content}]
                )
                
                # Parse JSON output
                res_text = response.content[0].text
                # strip code fences if present
                if "```json" in res_text:
                    res_text = res_text.split("```json")[1].split("```")[0]
                elif "```" in res_text:
                    res_text = res_text.split("```")[1].split("```")[0]
                return json.loads(res_text.strip())
                
            except Exception as e:
                print(f"[Guidance Error] Claude call failed, falling back to templates: {e}")
                
        # Return fallback rule-based templates
        return self._get_template_guidance(modality, findings_text)

    def _get_template_guidance(self, modality: str, findings_text: str) -> dict:
        """Rule-based clinical guidance compilation matching roadmap parameters."""
        f_lower = findings_text.lower()
        
        if modality == 'CT':
            return {
                'diet': {
                    'recommended': [
                        "Omega-3 rich foods (wild salmon, walnuts, chia seeds) for neural membrane support",
                        "High-antioxidant vegetables (berries, dark leafy greens) to counter oxidative stress",
                        "Whole grains and fiber to support glycemic stability"
                    ],
                    'avoid': [
                        "High sodium intake (limit to <1,500 mg per day) to control intracranial pressure and blood pressure",
                        "Saturated fats and processed meats that contribute to arterial plaque accumulation",
                        "Added sugars and refined carbohydrates"
                    ],
                    'reasoning': "Dietary focus aims to manage hypertension and provide neuroprotective nutrients during the acute/subacute recovery window."
                },
                'exercise': {
                    'allowed': [
                        "Absolute bed rest in the immediate acute phase",
                        "Semi-recumbent positioning with the head elevated at 30 degrees to minimize intracranial pressure",
                        "Very light, supervised walking once cleared by your neurologist"
                    ],
                    'restrictions': [
                        "Avoid all forms of strenuous workouts, heavy lifting, or bending over",
                        "Do not engage in valsalva maneuvers (straining)"
                    ],
                    'guidelines': "Physical activity must start at zero and scale up progressively under physical therapy supervision once stable."
                },
                'lifestyle': {
                    'tips': [
                        "Ensure consistent, high-quality sleep (7-8 hours) to promote brain recovery",
                        "Monitor blood pressure daily, noting any spikes",
                        "Ensure high hydration levels to maintain cerebral perfusion"
                    ],
                    'red_flags': [
                        "Sudden weakness or numbness in the face, arm, or leg (especially on one side)",
                        "Sudden confusion, slurred speech, or difficulty understanding speech",
                        "Sudden vision changes or loss of balance"
                    ],
                    'notes': "Immediate crisis management is critical. If any stroke red flags occur, contact emergency services immediately."
                },
                'routing': {
                    'specialist': "Neurologist & Neuroradiologist",
                    'urgency': "High to Immediate",
                    'guidance': "Take scan files and report directly to a stroke unit or emergency neurologist for clinical evaluation."
                },
                'questions': [
                    "Does this scan show signs of acute ischemia or older, chronic changes?",
                    "Is there any swelling (edema) or midline shifts that require therapy?",
                    "Should we perform a follow-up DWI-MRI scan for higher resolution?",
                    "Are there signs of vascular stenosis that need medication adjustments?"
                ]
            }
            
        elif modality == 'CXR':
            is_cardiomegaly = 'ctr' in f_lower or 'cardiomegaly' in f_lower
            
            if is_cardiomegaly:
                return {
                    'diet': {
                        'recommended': [
                            "Strict low-sodium foods (fresh fruits, vegetables, beans)",
                            "Potassium-rich items (bananas, sweet potatoes) to support myocardia",
                            "Lean protein (skinless chicken, tofu) and healthy monounsaturated oils"
                        ],
                        'avoid': [
                            "Salted nuts, canned soups, processed meats, and high-sodium seasonings",
                            "Excessive fluid intake (limit as advised by your physician to prevent volume overload)"
                        ],
                        'reasoning': "Managing sodium and fluid volume helps reduce blood pressure and cardiac workload."
                    },
                    'exercise': {
                        'allowed': [
                            "Low-impact aerobic walking (15-30 minutes daily) as tolerated",
                            "Gentle stretching and mobility exercises"
                        ],
                        'restrictions': [
                            "Avoid heavy weightlifting, push-ups, or exercises requiring holding your breath",
                            "Avoid sudden high-intensity bursts"
                        ],
                        'guidelines': "Activity should support circulation without placing sudden mechanical strain on ventricular tissue."
                    },
                    'lifestyle': {
                        'tips': [
                            "Weigh yourself daily; report sudden gains (>2 lbs in 24 hours) as they indicate fluid retention",
                            "Avoid NSAIDs (ibuprofen) which can worsen fluid retention",
                            "Prioritize stress reduction and regular sleep"
                        ],
                        'red_flags': [
                            "Sudden, severe shortness of breath, especially when lying flat",
                            "Chest pain, pressure, or radiative left-arm numbness",
                            "Swelling in the ankles, feet, or abdomen"
                        ],
                        'notes': "Heart size increases are secondary findings. A cardiologist must evaluate heart pump performance."
                    },
                    'routing': {
                        'specialist': "Cardiologist",
                        'urgency': "Moderate",
                        'guidance': "Consult cardiology for a diagnostic echocardiogram to evaluate ejection fraction (EF)."
                    },
                    'questions': [
                        "Does my enlarged heart silhouette correlate with high blood pressure or valve issues?",
                        "Should I undergo an echocardiogram to check my heart's pump performance?",
                        "What is my fluid intake limit per day?",
                        "Do I need medications like ACE-inhibitors or diuretics to protect my heart?"
                    ]
                }
            else: # Pneumonia / Consolidation default
                return {
                    'diet': {
                        'recommended': [
                            "Warm, hydrating broths, herbal teas, and high water volume to thin mucus",
                            "High-protein meals to aid tissue recovery",
                            "Vitamin C and zinc rich foods (citrus, spinach, pumpkin seeds)"
                        ],
                        'avoid': [
                            "Cold, sugary foods and carbonated drinks that can trigger coughing fits",
                            "Dairy products if they seem to increase mucus density",
                            "Processed snacks"
                        ],
                        'reasoning': "Hydration is essential to mobilize pulmonary secretions and clear consolidation zones."
                    },
                    'exercise': {
                        'allowed': [
                            "Complete physical rest during the fever/acute infection stage",
                            "Deep breathing exercises (5-10 repetitions hourly while awake)",
                            "Light, indoor walking once fever resolves"
                        ],
                        'restrictions': [
                            "Avoid aerobic running, gym workouts, or lifting",
                            "Avoid exposure to cold or dry air"
                        ],
                        'guidelines': "Focus on lung recruitment exercises (incentive spirometry) rather than cardiovascular training."
                    },
                    'lifestyle': {
                        'tips': [
                            "Use a humidifier to keep airway secretions thin",
                            "Avoid tobacco smoke, vaporizers, and chemical fumes",
                            "Elevate the head with pillows when sleeping to breathe easier"
                        ],
                        'red_flags': [
                            "Shortness of breath or breathing difficulty at rest",
                            "Oxygen saturation (SpO2) falling below 92%",
                            "High, persistent fever unresponsive to antipyretics"
                        ],
                        'notes': "Consolidation indicates fluid/cells in air spaces. Prompt diagnosis is needed to check for bacterial pneumonia."
                    },
                    'routing': {
                        'specialist': "Pulmonologist / General Physician",
                        'urgency': "Moderate to High",
                        'guidance': "Review these findings with a pulmonologist to check the need for empirical antibiotic therapy."
                    },
                    'questions': [
                        "Does this consolidation point to bacterial pneumonia, and do I need antibiotics?",
                        "How long should I perform deep breathing exercises?",
                        "Should we follow up with a Chest X-ray in 4-6 weeks to ensure the opacity has resolved?",
                        "Do I need an oxygen concentrator or hospital monitoring?"
                    ]
                }
                
        elif modality == 'ECG':
            return {
                'diet': {
                    'recommended': [
                        "Magnesium-rich foods (almonds, spinach, black beans) to support rhythm stability",
                        "Potassium sources (bananas, avocados) to aid myocardial conduction",
                        "Unsweetened decaffeinated beverages"
                    ],
                    'avoid': [
                        "Caffeine, coffee, energy drinks, and tea",
                        "Over-the-counter decongestants containing pseudoephedrine",
                        "Alcohol and nicotine, which act as cardiac irritants"
                    ],
                    'reasoning': "Dietary adjustments focus on eliminating stimulants that trigger rhythm variances and cardiac tachycardias."
                },
                'exercise': {
                    'allowed': [
                        "Progressive outdoor walking (15-30 minutes) at a comfortable pace",
                        "Yoga and meditation to decrease autonomic stress"
                    ],
                    'restrictions': [
                        "Avoid high-intensity intervals (HIIT), heavy squats, or competitive sports",
                        "Stop immediately if dizziness or palpitations occur"
                    ],
                    'guidelines': "Heart rate should remain within light boundaries. Cardiologist clearance is needed before return-to-sport."
                },
                'lifestyle': {
                    'tips': [
                        "Prioritize stress management (mindfulness, breath control)",
                        "Avoid sudden temperature extremes",
                        "Keep a symptom journal mapping palpitations to activities"
                    ],
                    'red_flags': [
                        "Chest pain, tightness, or pressure radiating to the jaw/neck/left arm",
                        "Fainting (syncope) or severe, sudden lightheadedness",
                        "Persistent, racing heart rate at rest (>120 BPM) with shortness of breath"
                    ],
                    'notes': "ECG captures electrical loops. A cardiologist should correlate these traces with physical symptoms."
                },
                'routing': {
                    'specialist': "Cardiologist",
                    'urgency': "Moderate",
                    'guidance': "Book a consultation with cardiology to review wave intervals and check if Holter monitoring is required."
                },
                'questions': [
                    "Does my ECG tracing show signs of arrhythmia, blockages, or ischemia?",
                    "Do I need a 24-hour Holter monitor or stress test to evaluate these findings?",
                    "Could my medications (or beta-blockers) be causing this rate shift?",
                    "Is my conduction interval (QT/QRS) within normal physiological ranges?"
                ]
            }
            
        elif modality == 'BLOOD_TEST':
            is_sugar = 'glucose' in f_lower
            is_anemia = 'hemoglobin' in f_lower or 'anemia' in f_lower
            
            if is_sugar:
                return {
                    'diet': {
                        'recommended': [
                            "Low-glycemic index carbohydrates (broccoli, cauliflower, berries)",
                            "High-fiber options (steel-cut oats, lentils, quinoa)",
                            "Lean protein (chicken breast, egg whites) to stabilize insulin response"
                        ],
                        'avoid': [
                            "Refined sugars, white flour, pastries, and white rice",
                            "Fruit juices and sugary sodas",
                            "Saturated fats and processed items"
                        ],
                        'reasoning': "A low-carb, high-fiber intake slows glucose absorption and helps prevent insulin spikes."
                    },
                    'exercise': {
                        'allowed': [
                            "Light, post-meal walking (10-15 minutes) to help peripheral glucose uptake",
                            "Moderate aerobics (cycling, swimming) for 30 minutes daily"
                        ],
                        'restrictions': [
                            "Avoid working out if blood sugar is extremely high (>250 mg/dL with ketones)",
                            "Avoid long workouts without a quick-carb snack handy"
                        ],
                        'guidelines': "Regular aerobic activity increases insulin sensitivity and helps lower long-term HbA1c."
                    },
                    'lifestyle': {
                        'tips': [
                            "Monitor fasting and post-meal glucose levels as advised",
                            "Prioritize high hydration (water only)",
                            "Maintain a food and glucose log"
                        ],
                        'red_flags': [
                            "Symptoms of hyperglycemia: extreme thirst, frequent urination, fatigue, blurry vision",
                            "Symptoms of hypoglycemia (if on meds): shaking, sweating, confusion, palpitations"
                        ],
                        'notes': "Fasting blood sugar above 100 mg/dL indicates pre-diabetes, and above 126 mg/dL indicates diabetes."
                    },
                    'routing': {
                        'specialist': "General Physician / Endocrinologist",
                        'urgency': "Moderate",
                        'guidance': "Review results with your primary physician to check if HbA1c testing or metformin is required."
                    },
                    'questions': [
                        "Does my fasting glucose level indicate pre-diabetes or diabetes?",
                        "Should we run an HbA1c test to evaluate my average 3-month sugar levels?",
                        "Should I consult an endocrinologist or certified diabetes educator?",
                        "Are these glucose levels manageable with diet and exercise alone?"
                    ]
                }
            elif is_anemia:
                return {
                    'diet': {
                        'recommended': [
                            "Heme-iron sources (lean beef, chicken liver, fish) for high absorption",
                            "Non-heme iron (spinach, lentils, fortified cereals)",
                            "Vitamin C rich foods (citrus fruits, strawberries) paired with iron meals"
                        ],
                        'avoid': [
                            "Drinking tea or coffee during or immediately after meals (polyphenols inhibit iron absorption)",
                            "High-calcium foods or supplements taken at the same time as iron (calcium competes for absorption)"
                        ],
                        'reasoning': "Vitamin C doubles non-heme iron absorption, while tea/coffee inhibits it by up to 60%."
                    },
                    'exercise': {
                        'allowed': [
                            "Light stretching and low-intensity walks",
                            "Frequent rest periods during activities"
                        ],
                        'restrictions': [
                            "Avoid heavy weights, sprinting, or exhausting workouts while hemoglobin is low",
                            "Stop if lightheadedness occurs"
                        ],
                        'guidelines': "Scale down workout intensity to match your body's current lower oxygen-carrying capacity."
                    },
                    'lifestyle': {
                        'tips': [
                            "Rise slowly from lying or sitting positions to prevent orthostatic dizziness",
                            "Maintain good hydration",
                            "Review other blood markers like Ferritin and Iron Saturation"
                        ],
                        'red_flags': [
                            "Severe shortness of breath with minimal exertion",
                            "Chest pain or rapid heartbeat at rest",
                            "Fainting (syncope)"
                        ],
                        'notes': "Low hemoglobin indicates anemia. Iron deficiency is the most common cause but requires diagnostic confirmation."
                    },
                    'routing': {
                        'specialist': "General Physician / Hematologist",
                        'urgency': "Moderate",
                        'guidance': "Consult your physician to establish the cause of anemia before initiating high-dose iron supplements."
                    },
                    'questions': [
                        "Is my low hemoglobin caused by iron deficiency, vitamin deficiency, or something else?",
                        "Should we check my serum ferritin, B12, and folate levels?",
                        "Do you recommend a specific dose of oral iron supplements, and for how long?",
                        "What follow-up blood tests do we need to track recovery?"
                    ]
                }
            else: # Default normal/generic blood test
                return {
                    'diet': {
                        'recommended': [
                            "Soluble fibers (oats, apples, beans) to bind lipid molecules",
                            "A balanced diet rich in clean vegetables and whole grains",
                            "Adequate hydration"
                        ],
                        'avoid': [
                            "Added sugars, fried foods, and saturated fats",
                            "Excessive alcohol intake"
                        ],
                        'reasoning': "General dietary adjustments help maintain organ function and support metabolic health."
                    },
                    'exercise': {
                        'allowed': [
                            "Moderate-intensity walking or jogging (30 minutes daily)",
                            "Active stretching and core strength routines"
                        ],
                        'restrictions': [
                            "Avoid strenuous workouts if feeling fatigued or unwell"
                        ],
                        'guidelines': "Maintain a consistent daily workout routine to help overall metabolic markers."
                    },
                    'lifestyle': {
                        'tips': [
                            "Keep regular sleep schedules",
                            "Avoid prolonged sitting, take short walking breaks",
                            "Repeat lab draws annually for prevention checkups"
                        ],
                        'red_flags': [
                            "Sudden weight gains, severe fatigue, or persistent fever",
                            "Unexplained pain or swelling"
                        ],
                        'notes': "General preventative wellness guidelines. Take tests annually."
                    },
                    'routing': {
                        'specialist': "General Family Physician",
                        'urgency': "Routine",
                        'guidance': "Review findings during your annual wellness checkup to maintain a baseline."
                    },
                    'questions': [
                        "Are my laboratory metrics within normal reference bounds?",
                        "Do I need any lifestyle changes based on this general panel?",
                        "When should I repeat these blood draws?",
                        "Should we run a lipid profile or liver function panel next time?"
                    ]
                }
        else: # Default fallback template
            return self._get_template_guidance('CT', findings_text)

_generator = None

def get_guidance_generator() -> GuidanceGenerator:
    global _generator
    if _generator is None:
        _generator = GuidanceGenerator()
    return _generator
