import cv2
import pytesseract
from deep_translator import GoogleTranslator

# Função para determinar se uma cor é geralmente clara ou escura
def e_clara(cor):
    # Calcula a luminância percebida da cor
    return (0.299 * cor[2] + 0.587 * cor[1] + 0.114 * cor[0]) > 127.5

# Caminho para o executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# Carregar a imagem
imagem = cv2.imread('imagem2.jpg')

# Converter para escala de cinza para OCR
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Usar OCR para detectar texto e sua localização
dados = pytesseract.image_to_data(imagem_cinza, output_type=pytesseract.Output.DICT)

# Inicializar o tradutor
translator = GoogleTranslator(source='auto', target='en')

# Loop sobre cada palavra detectada
for i in range(len(dados['text'])):
    if float(dados['conf'][i]) > 60:
        (x, y, w, h) = (dados['left'][i], dados['top'][i], dados['width'][i], dados['height'][i])

        # Calcular a média de cor da área do retângulo
        area_do_retangulo = imagem[y:y+h, x:x+w]
        cor_media = area_do_retangulo.mean(axis=0).mean(axis=0)

        # Desenhar um retângulo com a cor média sobre a área do texto
        cv2.rectangle(imagem, (x, y), (x + w, y + h), cor_media, -1)

        # Traduzir o texto
        texto = dados['text'][i]
        texto_traduzido = translator.translate(texto)

        # Determinar a cor do texto (branco ou preto) baseada na cor média do fundo
        cor_texto = (0, 0, 0) if e_clara(cor_media) else (255, 255, 255)

        # Calcular a posição centralizada do texto no eixo vertical
        tamanho_texto, _ = cv2.getTextSize(texto_traduzido, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = x
        text_y = y + h // 2 + tamanho_texto[1] // 2

        # Ajustar a posição e o tamanho da fonte para o texto traduzido
        cv2.putText(imagem, texto_traduzido, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, cor_texto, 2)

# Salvar ou mostrar a imagem resultante
cv2.imshow('Imagem Resultante', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()