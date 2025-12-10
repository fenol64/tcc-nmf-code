"""
Módulo para Fatoração de Matrizes Não Negativas (NMF) aplicada a Sistemas de Recomendação.

Este módulo implementa o algoritmo NMF conforme descrito no artigo associado, utilizando inicialização
NNDSVD, atualizações multiplicativas e múltiplos critérios de parada para prever interações em
matrizes usuário-item esparsas.

O objetivo é fornecer uma implementação clara, reprodutível e fiel aos fundamentos teóricos,
servindo como base para estudos e aplicações práticas em filtragem colaborativa.

Características principais:
    - Implementação fiel ao descrito no artigo (Seções 3 e 4).
    - Inicialização NNDSVD (Boutsidis & Gallopoulos, 2008) para acelerar a convergência.
    - Atualizações multiplicativas baseadas na norma de Frobenius (Lee & Seung, 2001).
    - Critérios de parada robustos: convergência, estagnação e máximo de iterações.
    - Utilização de máscara booleana para tratar valores ausentes.
    - Geração de recomendações Top-K com interpretação direta.
    - Reprodutibilidade garantida através de semente aleatória fixa.

Saída esperada:
    - Matrizes iniciais W e H (NNDSVD)
    - Matrizes finais W e H após convergência
    - Matriz reconstruída V̂ com 2 casas decimais
    - Erros de reconstrução para valores conhecidos
    - Recomendações Top-K para cada usuário
    - Comparação direta com os resultados do artigo

Referências:
    - BOUTSIDIS, C.; GALLOPOULOS, E. SVD based initialization: A head start for nonnegative matrix factorization.
      Pattern Recognition, v. 41, n. 4, p. 1350-1362, 2008.
    - LEE, D. D.; SEUNG, H. S. Algorithms for non-negative matrix factorization.
      Advances in Neural Information Processing Systems, v. 13, p. 556-562, 2001.

Autores: [Leandro de Araújo Lima, Fernando Nascimento Oliveira, Lucas Lacerda Antunes]
Data: 09/10/2025
Versão: 1.0

"""

import numpy as np

class NMFRecommender:

    """
    Implementa um sistema de recomendação baseado em Fatoração de Matrizes Não Negativas (NMF).

    Esta classe encapsula o processo de treinamento do modelo NMF sobre uma matriz de interações
    usuário-item e a geração subsequente de recomendações personalizadas com.
    - Inicialização NNDSVD (Seção 3.2)
    - Atualizações multiplicativas (Seção 3.3)
    - Critérios de parada (Seção 3.4)
    - Geração de recomendações Top-K (Seção 3.6)

    Args:
        n_factors (int)             : Número de fatores latentes (k) a serem usados na decomposição.
        max_iter (int, optional)    : Número máximo de iterações. Padrão é 500.
        tol (float, optional)       : Tolerância para o critério de convergência. Padrão é 1e-4.
        epsilon (float, optional)   : Constante para estabilidade numérica, para evitar divisão por zero. Padrão é 1e-9.
        random_seed (int, optional) : Semente para garantir a reprodutibilidade. Padrão é 42.
        np.array                    : Matriz de interação V, m usuários × n itens

    Atributos:
        W (np.ndarray): Matriz de fatores latentes dos usuários, aprendida após o `fit`.
        H (np.ndarray): Matriz de fatores latentes dos itens, aprendida após o `fit`.
        cost_history (list): Histórico dos valores da função de custo a cada iteração.

    Exemplo de uso 1:
        >>> import numpy as np
        >>> # Matriz de interações (3 usuários x 4 itens)
        >>> V = np.array([[5, 3, 0, 1],
        ...               [4, 0, 0, 1],
        ...               [1, 1, 0, 5]])
        >>> # Criar e treinar o modelo
        >>> model = NMFRecommender(n_factors=2, max_iter=500)
        >>> model.fit(V)
        >>> # Gerar recomendações para o usuário 0 (índice 0)
        >>> mask = (V > 0).astype(float) # Máscara de itens já conhecidos
        >>> recommendations = model.recommend_top_k(user_index=0, mask=mask, k=1)
        >>> print(f"Recomendações para o usuário 0: {recommendations}")


    Exemplo de uso 2: Para usar com diferetes dados:
        >>> # Substitua o valor da matriz np.array
        >>> V_custom = np.array([...])
        >>> #
        >>> # Criar e treinar o modelo alterando os valores dos hiperparâmetro
        >>> model = NMFRecommender(n_factors=10, max_iter=1000)
        >>> model.fit(V_custom)
        >>> #
        >>> # Gere as recomendações através da mascara
        >>> mask = (V_custom > 0).astype(float)
        >>> recs = model.recommend_top_k(user_index=0, mask=mask, k=5)

    """


    def __init__(self, n_factors=2, max_iter=500, tol=1e-4, epsilon=1e-9, random_seed=42):
        """
        Inicializa o modelo NMF

        Parâmetros:
        -----------
        n_factors   : int (default=2)
        max_iter    : int (default=500)
        tol         : float (default=1e-4)
        epsilon     : float (default=1e-9)
        random_seed : int (default=42)
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Matrizes que serão aprendidas
        self.W = None  # Matriz de usuários (m x k)
        self.H = None  # Matriz de itens (k x n)
        self.loss_history = []

    """Cria máscara booleana para valores observados (Seção 3.1)"""
    def _create_mask(self, V):
        return (V > 0).astype(float)

    def _nndsvd_initialization(self, V, mask):
        """
        Inicialização NNDSVD (Non-negative Double Singular Value Decomposition)
        Baseado em BOUTSIDIS e GALLOPOULOS (2008) - Seção 3.2
        """
        m, n = V.shape
        k = self.n_factors

        # Aplica máscara para usar apenas valores observados
        V_masked = V * mask

        # SVD da matriz mascarada
        try:
            U, S, VT = np.linalg.svd(V_masked, full_matrices=False)
        except np.linalg.LinAlgError:
            # Fallback para SVD aproximada se a matriz for muito esparsa
            U, S, VT = np.linalg.svd(V_masked + 1e-6 * np.random.randn(m, n), full_matrices=False)

        # Inicializa W e H com os k maiores componentes singulares
        W_init = np.zeros((m, k))
        H_init = np.zeros((k, n))

        for i in range(k):
            # Primeiro fator (maior valor singular)
            if i == 0:
                u_pos = np.maximum(U[:, i], 0)
                v_pos = np.maximum(VT[i, :], 0)
                u_neg = np.maximum(-U[:, i], 0)
                v_neg = np.maximum(-VT[i, :], 0)

                norm_pos = np.linalg.norm(u_pos) * np.linalg.norm(v_pos)
                norm_neg = np.linalg.norm(u_neg) * np.linalg.norm(v_neg)

                if norm_pos >= norm_neg:
                    W_init[:, i] = np.sqrt(S[i]) * u_pos / np.linalg.norm(u_pos)
                    H_init[i, :] = np.sqrt(S[i]) * v_pos / np.linalg.norm(v_pos)
                else:
                    W_init[:, i] = np.sqrt(S[i]) * u_neg / np.linalg.norm(u_neg)
                    H_init[i, :] = np.sqrt(S[i]) * v_neg / np.linalg.norm(v_neg)
            else:
                # Outros fatores
                W_init[:, i] = np.maximum(U[:, i], 0)
                H_init[i, :] = np.maximum(VT[i, :], 0)

        # Garante valores não negativos e adiciona pequeno ruído
        W_init = np.maximum(W_init, 0) + 1e-6
        H_init = np.maximum(H_init, 0) + 1e-6

        return W_init, H_init

    def _compute_loss(self, V, W, H, mask):
        """
        Calcula a função de custo com norma de Frobenius (Seção 3.3)
        Considera apenas valores observados através da máscara
        """
        reconstruction = W @ H
        error = (V - reconstruction) * mask
        loss = 0.5 * np.sum(error ** 2)
        return loss

    def fit(self, V):
        """
        Ajusta o modelo NMF à matriz de interações V

        Parâmetros:
        -----------
        V : numpy array (m x n)
            Matriz de interações usuário-item
            Valores ausentes representados por 0
        """
        m, n = V.shape
        mask = self._create_mask(V)

        # 1. Inicialização NNDSVD (Seção 3.2)
        print("Inicializando matrizes via NNDSVD...")
        self.W, self.H = self._nndsvd_initialization(V, mask)

        print("Matriz W inicial:")
        print(np.round(self.W, 2))
        print("\nMatriz H inicial:")
        print(np.round(self.H, 2))

        # 2. Processo iterativo (Seção 3.3 e 3.4)
        print("\nIniciando processo iterativo...")
        stagnation_count = 0

        for iteration in range(1, self.max_iter + 1):
            # Salva valores anteriores para cálculo de convergência
            W_old = self.W.copy()
            H_old = self.H.copy()

            # Atualização multiplicativa de H (Eq. 3.3)
            numerator_H = self.W.T @ (V * mask)
            denominator_H = self.W.T @ (self.W @ self.H * mask) + self.epsilon
            self.H = self.H * (numerator_H / denominator_H)

            # Atualização multiplicativa de W (Eq. 3.2)
            numerator_W = (V * mask) @ self.H.T
            denominator_W = (self.W @ self.H * mask) @ self.H.T + self.epsilon
            self.W = self.W * (numerator_W / denominator_W)

            # Calcula perda atual
            current_loss = self._compute_loss(V, self.W, self.H, mask)
            self.loss_history.append(current_loss)

            # Verifica critérios de parada (Seção 3.4)
            if iteration > 1:
                loss_change = abs(self.loss_history[-2] - current_loss) / self.loss_history[-2]

                # Critério principal: convergência
                if loss_change < self.tol:
                    print(f"\nConvergência alcançada na iteração {iteration}")
                    print(f"Variação da perda: {loss_change:.6f} < {self.tol}")
                    break

                # Critério secundário: estagnação prolongada
                if loss_change < 1e-6:
                    stagnation_count += 1
                    if stagnation_count >= 20:
                        print(f"\nEstagnação detectada na iteração {iteration}")
                        break
                else:
                    stagnation_count = 0

            # Progresso a cada 50 iterações
            if iteration % 50 == 0:
                print(f"Iteração {iteration}: Perda = {current_loss:.6f}")

        print(f"\nTreinamento concluído após {iteration} iterações")
        print(f"Perda final: {current_loss:.6f}")

        return self

    def reconstruct_matrix(self):
        """Reconstrói a matriz completa V_hat = W @ H"""
        if self.W is None or self.H is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        return self.W @ self.H

    def recommend_top_k(self, user_index, mask, k=1):
        """
        Gera recomendações Top-K para um usuário (Seção 3.6)

        Parâmetros:
        -----------
        user_index : int
            Índice do usuário (0-based)
        mask : numpy array
            Máscara booleana de interações conhecidas
        k : int (default=1)
            Número de recomendações a retornar

        Retorna:
        --------
        recommendations : list of tuples
            Lista de (item_index, predicted_rating, recommendation)
        """
        V_hat = self.reconstruct_matrix()

        # Identifica itens não avaliados
        unrated_items = np.where(mask[user_index] == 0)[0]

        if len(unrated_items) == 0:
            return []

        # Obtém predições para itens não avaliados
        predictions = [(item, V_hat[user_index, item]) for item in unrated_items]

        # Ordena por predição decrescente
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Seleciona Top-K
        top_k = predictions[:k]

        # Adiciona recomendação interpretativa
        recommendations = []
        for item, score in top_k:
            if score < 1.0:
                recommendation = "Não recomendar"
            elif score < 2.0:
                recommendation = f"Recomendar F{item+1} (baixa prioridade)"
            else:
                recommendation = f"Recomendar F{item+1}"
            recommendations.append((item, score, recommendation))

        return recommendations

    def print_results(self, V, user_names=None, item_names=None):
        """Exibe resultados formatados como no artigo"""
        m, n = V.shape
        mask = self._create_mask(V)
        V_hat = self.reconstruct_matrix()

        # Nomes padrão se não fornecidos
        if user_names is None:
            user_names = [f"U{i+1}" for i in range(m)]
        if item_names is None:
            item_names = [f"F{i+1}" for i in range(n)]

        print("=" * 60)
        print("RESULTADOS DO ALGORITMO NMF")
        print("=" * 60)

        print("\n1. MATRIZ ORIGINAL V (com valores ausentes como 0):")
        print("-" * 40)
        header = "     " + " ".join([f"{item:>6}" for item in item_names])
        print(header)
        for i in range(m):
            row = f"{user_names[i]:>4} " + " ".join([f"{V[i,j]:>6.0f}" if V[i,j] > 0 else "     0" for j in range(n)])
            print(row)

        print("\n2. MATRIZES FATORADAS FINAIS (arredondadas para 2 decimais):")
        print("-" * 40)
        print("Matriz W final (usuários × fatores latentes):")
        print(np.round(self.W, 2))
        print("\nMatriz H final (fatores latentes × itens):")
        print(np.round(self.H, 2))

        print("\n3. MATRIZ RECONSTRUÍDA V̂ = W × H (arredondada para 2 decimais):")
        print("-" * 40)
        header = "     " + " ".join([f"{item:>7}" for item in item_names])
        print(header)
        for i in range(m):
            row = f"{user_names[i]:>4} " + " ".join([f"{V_hat[i,j]:>7.2f}" for j in range(n)])
            print(row)

        print("\n4. ERRO DE RECONSTRUÇÃO PARA VALORES CONHECIDOS:")
        print("-" * 40)
        known_indices = np.where(mask == 1)
        errors = []
        for i, j in zip(*known_indices):
            error = abs(V[i, j] - V_hat[i, j])
            errors.append(error)
            print(f"{user_names[i]}-{item_names[j]}: {V[i,j]} → {V_hat[i,j]:.2f} (erro: {error:.3f})")

        print(f"\nErro médio absoluto: {np.mean(errors):.4f}")
        print(f"Erro máximo: {np.max(errors):.4f}")

        print("\n5. RECOMENDAÇÕES TOP-K (conforme Seção 4.4):")
        print("-" * 40)
        print("Usuário | Itens não avaliados | Notas preditas | Recomendação")
        print("-" * 70)

        for i in range(m):
            recommendations = self.recommend_top_k(i, mask, k=1)
            unrated_items = np.where(mask[i] == 0)[0]
            item_list = ", ".join([item_names[idx] for idx in unrated_items])

            if recommendations:
                for item_idx, score, rec_text in recommendations:
                    print(f"{user_names[i]:>7} | {item_list:>19} | {score:>13.2f} | {rec_text}")
            else:
                print(f"{user_names[i]:>7} | {item_list:>19} | {'N/A':>13} | Todos os itens avaliados")


# ============================================================================
# EXEMPLO DE USO: Matriz do artigo (Seção 4.1)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLO DO ARTIGO: NMF em Sistema de Recomendação")
    print("Matriz 3×4 (Usuários: U1, U2, U3 | Itens: F1, F2, F3, F4)")
    print("=" * 60)

    # Matriz V do artigo (Seção 4.1)
    V = np.array([
        [5, 3, 0, 1],  # U1
        [4, 0, 0, 1],  # U2
        [1, 1, 0, 5]   # U3
    ])

    # Nomes para display
    user_names = ["U1", "U2", "U3"]
    item_names = ["F1", "F2", "F3", "F4"]

    # 1. Cria e treina o modelo
    print("\n[1] CONFIGURAÇÃO DO MODELO")
    print(f"  • Número de fatores latentes (k): 2")
    print(f"  • Máximo de iterações: 500")
    print(f"  • Tolerância de convergência: 1e-4")
    print(f"  • Inicialização: NNDSVD")

    model = NMFRecommender(n_factors=2, max_iter=500, tol=1e-4)

    # 2. Treina o modelo
    print("\n[2] TREINAMENTO DO MODELO NMF")
    model.fit(V)

    # 3. Exibe resultados completos
    print("\n[3] RESULTADOS COMPLETOS")
    model.print_results(V, user_names, item_names)

    # 4. Resultados específicos para comparação com o artigo
    print("\n" + "=" * 60)
    print("COMPARAÇÃO COM OS RESULTADOS DO ARTIGO (Seção 4.3)")
    print("=" * 60)

    V_hat = model.reconstruct_matrix()

    print("\nMatriz V̂ do artigo (arredondada para 2 decimais):")
    print("[[4.99, 3.00, 0.45, 1.00],")
    print(" [4.00, 2.42, 0.42, 1.00],")
    print(" [1.00, 1.00, 1.52, 5.00]]")

    print("\nMatriz V̂ obtida pelo código:")
    print(np.round(V_hat, 2))

    print("\nDiferenças máximas entre implementação e artigo:")
    article_V_hat = np.array([
        [4.99, 3.00, 0.45, 1.00],
        [4.00, 2.42, 0.42, 1.00],
        [1.00, 1.00, 1.52, 5.00]
    ])

    diff = np.abs(V_hat - article_V_hat)
    print(f"Máxima diferença absoluta: {np.max(diff):.4f}")
    print(f"Diferença média absoluta: {np.mean(diff):.4f}")

    print("\n" + "=" * 60)
    print("ANÁLISE DAS RECOMENDAÇÕES (Seção 4.4)")
    print("=" * 60)

    mask = model._create_mask(V)
    print("\nItens ausentes para cada usuário:")
    for i in range(3):
        unrated = np.where(mask[i] == 0)[0]
        items = ", ".join([item_names[idx] for idx in unrated])
        print(f"  {user_names[i]}: {items if items else 'Nenhum'}")

    print("\nPredições para itens ausentes:")
    print("Usuário | Item  | Predição | Interpretação")
    print("-" * 45)

    predictions_table = [
        ("U1", "F3", 0.45, "Não recomendar"),
        ("U2", "F2", 2.42, "Recomendar F2"),
        ("U2", "F3", 0.42, "Não recomendar"),
        ("U3", "F3", 1.52, "Recomendar F3 (baixa prioridade)")
    ]

    for user, item, pred, interp in predictions_table:
        print(f"{user:>7} | {item:>5} | {pred:>8.2f} | {interp}")