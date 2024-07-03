import ollama
from utils.llm_interaction import gen_refined_prompts

scene = "A cabin in the woods"
gen_refined_prompts(scene, 10)