import ollama

def gen_refined_prompts(scene_prompt, n):

    response = ollama.chat(model='llama3', messages=[
    {
      'role': 'user',
      'content': f'''
      Imagine and refine the scene : \'{scene_prompt}\' by adding elements or details, giving the scene a sense of layering and depth.
      Generate {n} local descriptions about the surroundings, evenly distributed in horizontal 360 degree, each description only contains content from that angle.
      You should write the description in format of: \n 1. <desc> desc A </desc>\n 2. <desc> desc B </desc>\n ...\n and only the description, no narratives or explicit directions
      The sentence style inside <desc></desc> should be similar to stable diffusion prompts.
      Each sentence must be a prompt of a whole scene, not just part of it, and must have the subject of the description, only describe the visible elements.
      Each description of a angle should have unique objects, the main object only explicitly occurred in one description.
      Each description shouldn't contain the parts from the same object.
      There shouldn't be a large gap in content between two adjacent descriptions of the scene.
      ''',
    },
    ])
    out_text : str = response['message']['content']

    descs = out_text.split('<desc>')
    res = []
    for desc in descs[1:]:
        res.append(desc.split('</desc>')[0].strip())

    print(res, len(res))
    return res