import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, AIMessage

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- 1. CONFIGURAÇÃO DA FERRAMENTA E DO AGENTE ---

# Inicializa o session_state para o dataframe
if "df" not in st.session_state:
    st.session_state.df = None

def carregar_dataframe(caminho_arquivo: str):
    """Carrega o dataframe no session_state."""
    try:
        df = pd.read_csv(caminho_arquivo)
        st.session_state.df = df
        st.session_state.dataframe_carregado = True
        print(f"DataFrame de {caminho_arquivo} carregado.")
        print(f"Colunas: {list(df.columns)}")
        print(f"Shape: {df.shape}")
    except Exception as e:
        st.error(f"Erro ao carregar o dataframe: {e}")

def python_repl_with_df(code: str) -> str:
    """Executa código Python com acesso ao dataframe."""
    try:
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Cria um namespace local com o dataframe
        local_vars = {
            "df": st.session_state.df,
            "pd": pd,
            "os": os
        }
        
        # Importa matplotlib se necessário
        if "plt" in code or "plot" in code or "hist" in code:
            import matplotlib.pyplot as plt
            local_vars["plt"] = plt
            plt.clf()
        
        # Captura a saída do print
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            # Tenta avaliar como expressão primeiro (para retornar valores)
            try:
                result = eval(code, local_vars)
                if result is not None:
                    print(result)
            except:
                # Se falhar, executa como statement
                exec(code, local_vars)
        
        # Pega o output capturado
        output = output_buffer.getvalue()
        
        # Se um gráfico foi criado, salva com nome único
        if "plt" in local_vars and local_vars["plt"] is not None:
            try:
                import matplotlib.pyplot as plt
                import time
                
                # Gera nome único com timestamp
                timestamp = int(time.time() * 1000)
                plot_name = f"plot_{timestamp}.png"
                
                plt.tight_layout()
                plt.savefig(plot_name, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Armazena o nome do último gráfico criado
                st.session_state.last_plot = plot_name
                
                if output:
                    output += f"\n[Gráfico salvo como {plot_name}]"
                else:
                    output = f"Gráfico salvo como {plot_name}"
            except Exception as e:
                output += f"\n[Erro ao salvar gráfico: {e}]"
        
        return output if output else "Código executado com sucesso (sem output)."
    
    except Exception as e:
        return f"Erro na execução: {str(e)}"

# Ferramenta Python personalizada
python_analyst_tool = Tool(
    name="python_repl_tool",
    description=(
        "Use esta ferramenta para executar código Python e analisar o dataframe 'df'. "
        "O dataframe já está carregado como 'df'. Você tem acesso a pandas (pd) e matplotlib.pyplot (plt). "
        "Para criar gráficos, use plt.figure(), faça o plot, e use plt.savefig('plot.png'). "
        "Exemplos: df.head(), df.info(), df.describe(), plt.hist(df['coluna']), etc."
    ),
    func=python_repl_with_df
)
tools = [python_analyst_tool]

# Puxa o prompt padrão para agentes do LangChain Hub
prompt = hub.pull("hwchase17/react")

# LLM que o agente usará (configurado para Gemini)
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    max_tokens=2048,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Cria o agente
agent = create_react_agent(llm, tools, prompt)

# Cria o AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
    max_execution_time=180
)

# --- 2. APLICAÇÃO STREAMLIT ---

st.title("🤖 Agente Autônomo para Análise de Dados")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframe_carregado" not in st.session_state:
    st.session_state.dataframe_carregado = False

uploaded_file = st.file_uploader("Faça o upload do seu arquivo CSV", type=["csv"])

# Cria o diretório de uploads se não existir
os.makedirs("uploads", exist_ok=True)

if uploaded_file and not st.session_state.dataframe_carregado:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    carregar_dataframe(file_path)
    
    if st.session_state.df is not None:
        st.success(f"✅ Arquivo '{uploaded_file.name}' carregado!")
        st.write(f"**Shape:** {st.session_state.df.shape}")
        st.write(f"**Colunas:** {', '.join(st.session_state.df.columns)}")

if st.session_state.dataframe_carregado:
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)
            if isinstance(message, AIMessage) and "image_path" in message.additional_kwargs:
                st.image(message.additional_kwargs["image_path"])

    if prompt_input := st.chat_input("Faça sua pergunta sobre os dados..."):
        st.session_state.messages.append(HumanMessage(content=prompt_input))
        with st.chat_message("human"):
            st.markdown(prompt_input)

        with st.chat_message("ai"):
            with st.spinner("O agente está pensando..."):
                try:
                    response = agent_executor.invoke({"input": prompt_input})
                    final_answer = response["output"]
                    
                    ai_message = AIMessage(content=final_answer)

                    # Verifica se um novo gráfico foi criado nesta execução
                    if hasattr(st.session_state, 'last_plot') and st.session_state.last_plot:
                        if os.path.exists(st.session_state.last_plot):
                            st.image(st.session_state.last_plot)
                            ai_message.additional_kwargs = {"image_path": st.session_state.last_plot}
                        # Limpa o último gráfico para não repetir
                        st.session_state.last_plot = None

                    st.markdown(final_answer)
                    st.session_state.messages.append(ai_message)

                except Exception as e:
                    st.error(f"Erro: {e}")
else:
    st.info("📁 Aguardando o upload de um arquivo CSV...")