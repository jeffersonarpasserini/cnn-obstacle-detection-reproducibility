# exploratory/post hoc — revisão do autor

Preencha cluster_names.csv e error_annotations.csv. Nomes de clusters e categorias estão vazios. Nenhum rótulo foi alterado. Os mosaicos de clusters não mostram classes ou nomes originais; mosaicos de consenso mostram verdade/previsão e identificador por hash.

Categorias permitidas: sombra, mudança de piso/textura, junta ou rachadura, grama/vegetação, faixa ou pintura, baixa luz, desfoque, obstáculo pequeno/distante, rótulo questionável, outro.

Consenso usa decisões padrão congeladas dos quatro fixos e maioria das seeds de EfficientNetB3. Confiança no erro: média dos quatro escores direcionados ao erro e da mediana das três seeds FT. Mostram-se no máximo 60 por tipo; o CSV inclui todos os erros de consenso. Acurácia-oráculo e atributos mais informativos são escolhas externas pós-teste, sem uso implantável.
