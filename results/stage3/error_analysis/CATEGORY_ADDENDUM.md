# Adendo às categorias de erro (exploratório/post hoc)

Data: 2026-10-09 (atualizado em 2026-10-13). Decisão do autor.

A lista original em `configs/stage3.json` (`error_categories`) continua como está, porque o arquivo
faz parte dos arquivos protegidos por hash da etapa 3 (`protected_before.json` / `protected_after.json`
e `config_sha256` em `registration.json`). As categorias abaixo foram acrescentadas só para a
anotação humana de `error_annotations.csv`. Elas não alteram modelos, limiares, predições nem rótulos.

## Categorias acrescentadas

| Categoria | Definição | Motivo |
|---|---|---|
| `bueiro/grelha` | Tampa de bueiro ou poço de visita, ou grelha de drenagem, rente ao piso e sem desnível evidente, em imagem rotulada como livre | É o padrão de erro mais frequente entre as livres classificadas como obstruídas. Antes estava em `outro`. |
| `piso tátil` | Sinalização tátil de alerta ou direcional no piso (para pessoas com deficiência visual), em imagem rotulada como livre | Elemento de acessibilidade confundido com obstáculo; relevante para o usuário-alvo. Incluída na revisão do autor de 2026-10-13. |
| `sem causa aparente` | Caminho livre, sem nenhum elemento visível que explique o erro (sem obstáculo, mudança de piso, sombra forte, baixa luz ou elemento no piso) | Registra separadamente os erros que não se explicam pelo conteúdo visível da imagem, em vez de diluí-los em `outro`. Incluída na revisão do autor de 2026-10-13. |
| `borda de piscina` | Borda de piscina (água, lona ou capa de proteção) junto ao caminho, em imagem rotulada como obstruída | Padrão principal das obstruídas classificadas como livres; corresponde ao grupo visual 5, quase ausente no VIA (51 externas contra 1 no VIA). Incluída na revisão do autor de 2026-10-13. |
| `corredor estreito` | Passagem estreita (corredor interno ou externo) com paredes ou objetos próximos dos dois lados e centro livre, em imagem rotulada como livre | Padrão recorrente de erro ligado ao grupo visual 7 (ambientes internos). Incluída na revisão do autor de 2026-10-13. |
| `cena aberta/horizonte` | Rua ou calçada ampla vista em perspectiva, com horizonte, céu e elementos distantes, sem obstáculo no caminho imediato | Corresponde ao grupo visual 2, sub-representado no VIA (188 externas contra 23 no VIA). Antes estava em `outro`. |

## Lista completa usada na anotação

sombra, mudança de piso/textura, junta ou rachadura, grama/vegetação, faixa ou pintura, baixa luz,
desfoque, obstáculo pequeno/distante, rótulo questionável, bueiro/grelha, cena aberta/horizonte, piso tátil, sem causa aparente, borda de piscina, corredor estreito, outro.

## Regras de precedência

- `bueiro/grelha` tem precedência sobre `sombra` e `mudança de piso/textura` quando a tampa ou a grelha é o elemento dominante.
- `cena aberta/horizonte` vale só quando não há elemento dominante no primeiro plano. Calçadas amplas sem horizonte continuam em `outro`.
- Faixa de pedestre com grelha fica em `faixa ou pintura`.
- `piso tátil` tem precedência sobre `mudança de piso/textura` e sobre veículos ou postes distantes quando a faixa tátil é o elemento dominante. Rampas com corrimão e piso tátil continuam em `rótulo questionável`.
- `sem causa aparente` só se aplica quando nenhuma outra categoria descreve a imagem. Elementos laterais fora do caminho (grama, muro, portão) não impedem o uso desta categoria.
- `outro` fica reservado para causas visíveis que não têm categoria própria (nenhum caso na revisão de 2026-10-13).
- `cena aberta/horizonte` inclui caminhos ou campos de terra amplos com horizonte (ex.: clear.0304, clear.0347). Trilhas de terra em close ficam em `mudança de piso/textura`.
- `corredor estreito` tem precedência sobre `rótulo questionável` quando o centro do caminho está livre e o objeto lateral não invade a passagem (ex.: clear.0192).
