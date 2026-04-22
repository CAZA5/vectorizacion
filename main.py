import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

# ---------------------------------------------------------
# 1. PREPARACIÓN DE DATOS (Asegúrate de tener tu corpus aquí)
# ---------------------------------------------------------
# Ejemplo por si no has cargado el corpus previo
if 'corpus_lematizado' not in locals():
    corpus_lematizado = [
        "el gato corre por el jardin",
        "el perro ladra en el jardin",
        "los pajaros vuelan sobre el jardin",
        "el gato y el perro son amigos",
        "la naturaleza es increible y verde"
    ]

# ---------------------------------------------------------
# 2. FUNCIÓN MEJORADA PARA GRAFICAR
# ---------------------------------------------------------
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, cmap_name='viridis'):
    # Transponer para analizar palabras (Filas = Palabras)
    matriz_palabras = matriz.T.toarray()
    
    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras)
    
    # Varianza explicada para el título (informativo)
    var_exp = np.sum(pca.explained_variance_ratio_) * 100
    
    # Colorear según la norma (distancia al origen) para resaltar relevancia
    normas = np.linalg.norm(matriz_palabras, axis=1)
    
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                    c=normas, cmap=cmap_name, s=100, 
                    edgecolors='white', linewidth=0.5, alpha=0.9)
    
    # Etiquetado inteligente
    for i, palabra in enumerate(vocabulario):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2] + 0.02, 
                palabra, fontsize=10, weight='semibold', alpha=0.8)
        
    ax.set_title(f"{titulo}\n(Varianza explicada: {var_exp:.1f}%)", pad=20)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    
    # Estética del grid
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    plt.colorbar(sc, ax=ax, shrink=0.5, label='Intensidad de frecuencia/valor')

# ---------------------------------------------------------
# 3. EJECUCIÓN Y COMPARATIVA
# ---------------------------------------------------------
fig = plt.figure(figsize=(20, 9))

# --- CONFIGURACIÓN DE VECTORIZADORES ---
# Añadimos stop_words en español para limpiar ruidos comunes
config_vec = {"stop_words": ['el', 'la', 'los', 'las', 'un', 'una', 'y', 'en', 'por', 'sobre']}

# A. BAG OF WORDS
ax1 = fig.add_subplot(121, projection='3d')
bow_vec = CountVectorizer(**config_vec)
X_bow = bow_vec.fit_transform(corpus_lematizado)
graficar_palabras_3d(ax1, X_bow, bow_vec.get_feature_names_out(), "BoW (Frecuencia Absoluta)", 'Wistia')

# B. TF-IDF
ax2 = fig.add_subplot(122, projection='3d')
tfidf_vec = TfidfVectorizer(**config_vec)
X_tfidf = tfidf_vec.fit_transform(corpus_lematizado)
graficar_palabras_3d(ax2, X_tfidf, tfidf_vec.get_feature_names_out(), "TF-IDF (Importancia Relativa)", 'Viridis')

plt.tight_layout()
plt.show()