import pandas as pd

# Carrega o CSV enviado
df = pd.read_csv("HandInfo - HandInfo2.csv")

# Corrige nomes de coluna
df.columns = [col.strip() for col in df.columns]

# Filtra homens
homens = df[(df['gender'] == 'male')].copy()
homens['label'] = 0

# Filtra mulheres
mulheres = df[df['gender'] == 'female'].copy()
mulheres['label'] = 1

# Junta os dois grupos
df_final = pd.concat([homens, mulheres])

# Corrigido: usa 'imageName' em vez de 'fileName'
df_final[['imageName', 'label']].to_csv("maos_reclassificadas2.csv", index=False)

print("✅ Novo CSV salvo como 'maos_reclassificadas2.csv'")
