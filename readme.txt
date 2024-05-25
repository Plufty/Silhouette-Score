1 - Para a execução do código primeiro deverão ser instalados os módulos numpy, pandas, datetime e matplotlib.
2 - O código foi comentado de forma a demonstrar onde foi cumprida cada uma das condições impostas na tarefa.
3 - Como os códigos do k-means e single link já haviam sido antes implementados, foram apenas copiados das práticas anteriores e removidos seus arquivos de saída desnecessários para esta prática.
4 - O arquivo csv/data de entrada deve ser altetrado na linha 129 do código.
5 - O Arquivo csv/data deverá conter a classe como a última coluna do arquivo, pois esta será ignorada na execução do algoritmo.
6 - Os dados são carregados e normalizados para os cálculos.
7 - A função single_link junto com as funções criadas para auxiliá-la fazem o cálculo dos clusters utilizando single_link. (Lembrando, este não está clusterizando com objetivo de fazer clusters equilibrados, mas sim no final ter apenas um cluster, tendo isso em vista ele pode gerar clusters estranhos, como por exemplo utilizando o dataset iris com K=3, ele gera um cluster com um único elemento e outros dois gigantes)
8 - A classe KMeans junto com funções criadas para auxiliá-la fazem o agrupamento utilizando k-means. 
9 - A função calculate_silhouette é utilizada para calcular a silhueta simplificada de acordo com a fórmula vista em sala de aula.
10 - O usuário deverá informar o número de clusters, a função input_int garante que seja um número inteiro válido para gerar os clusters.
11 - Ao fim do código será gerados apenas um arquivo com nome "resultado" seguido de data e hora em que ele foi gerado e nele estará contido os clusters finais de cada algoritmo, além do silhouettte score de cada algoritmo e qual apresentou o melhor resultado.