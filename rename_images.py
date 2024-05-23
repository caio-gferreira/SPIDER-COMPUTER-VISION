import os
from PIL import Image

def renomear_imagens(pasta):
    contador = 1
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".jpg") or arquivo.endswith(".jpeg") or arquivo.endswith(".png"):
            caminho_antigo = os.path.join(pasta, arquivo)
            extensao = os.path.splitext(arquivo)[1].lower()
            novo_nome = "formiga" + str(contador) + extensao
            caminho_novo = os.path.join(pasta, novo_nome)
            os.rename(caminho_antigo, caminho_novo)
            contador += 1

if __name__ == "__main__":
    pasta = input("Digite o caminho da pasta contendo as imagens: ")
    renomear_imagens(pasta)
    print("Imagens renomeadas com sucesso!")
