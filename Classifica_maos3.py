import pandas as pd

# Carrega o CSV 'HandInfo3.csv'
df = pd.read_csv("HandInfo3.csv")

# Corrige nomes de coluna
df.columns = [col.strip() for col in df.columns]

# Filtra mãos sem acessório (coluna 'accessories' == 0)
sem_acessorio = df[df['accessories'] == 0].copy()
sem_acessorio['label'] = 0

# Filtra mãos com acessório (coluna 'accessories' == 1)
com_acessorio = df[df['accessories'] == 1].copy()
com_acessorio['label'] = 1

# Junta os dois grupos
df_final = pd.concat([sem_acessorio, com_acessorio])

# Salva apenas as colunas 'imageName' e 'label'
df_final[['imageName', 'label']].to_csv("maos_reclassificadas3.csv", index=False)

print("✅ Novo CSV salvo como 'maos_reclassificadas3.csv'")