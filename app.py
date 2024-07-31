from flask import Flask, Response, jsonify, request, stream_with_context
import numpy as np

import json
 
app = Flask(__name__)

def multiplicar_matrices(A, B):
    return np.dot(A, B)

def transponer(A):
    return np.transpose(A)

def sumar_matrices(A, B):
    return A + B

def restar_matrices(A, B):
    return A - B

def invertir_matriz(A):
    return np.linalg.inv(A)

@app.route('/')
def hello_world():
    return 'Hello, World!'


def convertir_a_lista(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convertir_a_lista(i) for i in data]
    elif isinstance(data, dict):
        return {key: convertir_a_lista(value) for key, value in data.items()}
    else:
        return data


@app.route('/api/data', methods=['POST'])
def post_data():
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    x = data["niveles"]
    y = data["vanos"]

    X = x * (y - 1)
    Y = x * (y + 1)
    NB = (y + 1) * x + x * y
    GLG = (y + 1) * x * 3
    GLL = NB * 6
    

    Ec = data['elasticidad']
    Pc = data['poisson']
    
    carga = data['carga']

    columnasRigidez = []
    for i, columna in enumerate(data['columnas']):
        Lc = columna['longitud']
        Bc = columna['ancho']
        Hc = columna['peralte']
        Mc = Ec / (2 * (Pc + 1))
        Ic = (Bc * Hc ** 3) / 12
        A = Bc * Hc
        AC = A * 0.83333333333
        ALFAc = (12 * Ec * Ic) / (Lc ** 2 * Mc * AC)
        ac = ALFAc + 1

        D12c = (12 * Ec * Ic) / (ac * Lc ** 3)
        D6c = (6 * Ec * Ic) / (ac * Lc ** 2)
        D4c = ((4 + ALFAc) * Ec * Ic) / (ac * Lc)
        D2c = ((2 - ALFAc) * Ec * Ic) / (ac * Lc)
        D1c = (Ec * A) / Lc

        Kc = np.array([
            [D12c, -D12c, -D6c, -D6c, 0, 0],
            [-D12c, D12c, D6c, D6c, 0, 0],
            [-D6c, D6c, D4c, D2c, 0, 0],
            [-D6c, D6c, D2c, D4c, 0, 0],
            [0, 0, 0, 0, D1c, -D1c],
            [0, 0, 0, 0, -D1c, D1c],
        ])

        columnasRigidez.append({'Kc': Kc, 'columnaIndex': i + 1})

    print(columnasRigidez)
    
    Lv = data['viga']['longitud']
    Bv = data['viga']['ancho']
    Hv = data['viga']['peralte']
    Ev = Ec
    Gv = Ev / (2 * (Pc + 1))
    Iv = (Bv * Hv ** 3) / 12
    Av = Bv * Hv
    AV = Av * 0.83333333333
    ALFAv = (12 * Ev * Iv) / (Lv ** 2 * Gv * AV)
    av = ALFAv + 1

    D12v = (12 * Ev * Iv) / (av * Lv ** 3)
    D6v = (6 * Ev * Iv) / (av * Lv ** 2)
    D4v = ((4 + ALFAv) * Ev * Iv) / (av * Lv)
    D2v = ((2 - ALFAv) * Ev * Iv) / (av * Lv)
    D1v = (Ev * Av) / Lv

    Kv = np.array([
        [D12v, -D12v, -D6v, -D6v, 0, 0],
        [-D12v, D12v, D6v, D6v, 0, 0],
        [-D6v, D6v, D4v, D2v, 0, 0],
        [-D6v, D6v, D2v, D4v, 0, 0],
        [0, 0, 0, 0, D1v, -D1v],
        [0, 0, 0, 0, -D1v, D1v],
    ])
    print("------ viga")
    print(Kv)
    
    # Definir las dimensiones y otros parámetros necesarios

    # Crear una matriz de ceros con dimensiones Y x NB
    Matriz_M = np.zeros((Y, NB))

    # Matriz de correlación para vigas
    i = 0  # Dato auxiliar para salto de barra
    t = 1  # Dato auxiliar para salto de piso

    for barra in range(NB, Y, -1):  # Recorrer las vigas de NB a Y+1
        Matriz_M[t - 1 + x * i][barra - 1] = 1
        Matriz_M[t - 1 + x * (i + 1)][barra - 1] = 2
        i += 1
        if i == y:
            i = 0
            t += 1

    # Matriz de correlación para columnas
    i = 0  # Dato auxiliar para salto de barra
    t = 1  # Dato auxiliar para salto de piso

    for barra in range(Y, 0, -1):  # Recorrer las barras de Y a 1
        Matriz_M[t - 1 + x * i][barra - 1] = 1
        if barra > y + 1:  # Condición para las primeras columnas
            Matriz_M[t + x * i][barra - 1] = 2
        i += 1
        if i == y + 1:
            i = 0
            t += 1
    print("__________ matrix")
    print(Matriz_M)
    
    
    
    # Crear la matriz gradosDeLibertad con dimensiones NB x 3 x 2
    gradosDeLibertad = np.zeros((NB, 3, 2), dtype=int)

    impar = 1
    par = 2

    # Asignar valores a gradosDeLibertad
    for k in range(NB):
        for j in range(3):
            gradosDeLibertad[k][j][0] = impar
            gradosDeLibertad[k][j][1] = par
            impar += 2
            par += 2
            
    print("----grados")
    print(gradosDeLibertad)

    # Crear la matriz MGlobal con dimensiones GLL x GLG inicializada con ceros
    MGlobal = np.zeros((GLL, GLG))

    # Rellenar MGlobal basado en Matriz_M y gradosDeLibertad
    for i in range(Y):
        for j in range(NB):
            if Matriz_M[i][j] == 1:
                if j > Y - 1:
                    MGlobal[gradosDeLibertad[j][2][0] - 1][i] = 1
                else:
                    MGlobal[gradosDeLibertad[j][0][0] - 1][i] = 1
            elif Matriz_M[i][j] == 2:
                if j > Y - 1:
                    MGlobal[gradosDeLibertad[j][2][1] - 1][i] = 1
                else:
                    MGlobal[gradosDeLibertad[j][0][1] - 1][i] = 1

    for i in range(Y):
        for j in range(NB):
            if Matriz_M[i][j] == 1:
                if j > Y - 1:
                    MGlobal[gradosDeLibertad[j][0][0] - 1][i + Y] = -1
                else:
                    MGlobal[gradosDeLibertad[j][2][0] - 1][i + Y] = 1
            elif Matriz_M[i][j] == 2:
                if j > Y - 1:
                    MGlobal[gradosDeLibertad[j][0][1] - 1][i + Y] = -1
                else:
                    MGlobal[gradosDeLibertad[j][2][1] - 1][i + Y] = 1

    for i in range(Y):
        for j in range(NB):
            if Matriz_M[i][j] == 1:
                MGlobal[gradosDeLibertad[j][1][0] - 1][i + Y * 2] = 1
            elif Matriz_M[i][j] == 2:
                MGlobal[gradosDeLibertad[j][1][1] - 1][i + Y * 2] = 1
                
    print("-------- matrix global")
    print(MGlobal)
    
    # Crear las submatrices de MGlobal
    submatrices = [MGlobal[k*6:(k+1)*6] for k in range(NB)]

    t_aux = 1
    aportesBarras = []

    # Calcular aportes de las barras
    for i in range(NB):
        if i < Y:
            B = multiplicar_matrices(
                transponer(submatrices[i]),
                columnasRigidez[t_aux - 1]['Kc']
            )
            B = multiplicar_matrices(B, submatrices[i])
        else:
            B = multiplicar_matrices(transponer(submatrices[i]), Kv)
            B = multiplicar_matrices(B, submatrices[i])

        if (i + 1) % (y + 1) == 0:
            t_aux += 1

        aportesBarras.append(B)

    # Calcular rigidez global de la estructura
    Kglobal = np.zeros_like(aportesBarras[0])
    for k in range(NB):
        Kglobal = sumar_matrices(Kglobal, aportesBarras[k])

    # Calcular rigidez lateral de la estructura
    K1 = Kglobal[:x, :x]
    K2 = Kglobal[x:, :x]
    K3 = Kglobal[x:, x:]

    K3_inv = invertir_matriz(K3)
    KL = restar_matrices(
        K1,
        multiplicar_matrices(transponer(K2), multiplicar_matrices(K3_inv, K2))
    )

    # Calcular deformaciones laterales de la estructura
    M_Carga = np.full((x, 1), carga)
    Deformaciones = multiplicar_matrices(invertir_matriz(KL), M_Carga)

    # Resultado final
    resultado = {
        'columnasRigidez': columnasRigidez,
        'Kv': Kv,
        'Matriz_M': Matriz_M,
        'matrices': gradosDeLibertad,
        'MGlobal': MGlobal,
        'submatrices': submatrices,
        'aportesBarras': aportesBarras,
        'Kglobal': Kglobal,
        'KL': KL,
        'Deformaciones': Deformaciones,
    }
    print(Deformaciones)
    # Convertir todos los ndarrays a listas de Python
    resultado_json = convertir_a_lista(resultado)

    # Convertir los datos a una cadena de texto JSON
    json_data = json.dumps(resultado_json)
    
    return Response(
        json_data,
        mimetype='text/plain',
        headers={'Content-Disposition': 'attachment;filename=resultado.txt'}
    )

if (__name__ == '__main__'):
    app.run()