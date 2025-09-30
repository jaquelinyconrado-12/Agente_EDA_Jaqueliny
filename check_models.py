import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carrega a chave de API do arquivo .env
load_dotenv()

# Configura a API do Google
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("A variável GOOGLE_API_KEY não foi encontrada ou está vazia no arquivo .env")
    
    genai.configure(api_key=api_key)
    
    print("Buscando modelos disponíveis para sua chave de API...")
    
    model_found = False
    # Lista todos os modelos e imprime seus nomes se suportarem 'generateContent'
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            model_found = True

    if not model_found:
        print("\nNenhum modelo compatível encontrado. Verifique se sua chave de API está correta e tem as permissões necessárias no Google AI Studio.")
    else:
        print("\nCopie um dos nomes de modelo da lista acima (ex: 'models/gemini-pro') para usarmos em nosso agente.")

except Exception as e:
    print(f"\nOcorreu um erro: {e}")
    print("Verifique se sua chave GOOGLE_API_KEY está correta no arquivo .env e se a biblioteca 'google-generativeai' está instalada ('pip install google-generativeai').")