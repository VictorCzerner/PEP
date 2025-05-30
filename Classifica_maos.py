import pandas as pd

# Carrega o CSV enviado
df = pd.read_csv("HandInfo - HandInfo.csv")

# Corrige nomes de coluna
df.columns = [col.strip() for col in df.columns]

# Filtra adultos jovens (20 a 40 anos)
adultos = df[(df['age'] >= 20) & (df['age'] <= 40)].copy()
adultos['label'] = 0

# Filtra idosos (60+)
idosos = df[df['age'] >= 60].copy()
idosos['label'] = 1

# Junta os dois grupos
df_final = pd.concat([adultos, idosos])

# Corrigido: usa 'imageName' em vez de 'fileName'
df_final[['imageName', 'label']].to_csv("maos_reclassificadas.csv", index=False)

print("✅ Novo CSV salvo como 'maos_reclassificadas.csv'")
