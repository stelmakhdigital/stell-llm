# Отчёт потока models — раунд 6 ДОБОР (T-20261017200354)

Ветка `agent/models/8` (база — merge models-7 `048500a`; примечание: на
`origin` (GitHub) ветка `master` исчезла из ls-remote («gone» в локальном
remotes) — база взята с локального `master`, который содержит ровно
merge-коммит `048500a` раунда 6). Работы: поддержка `Qwen3MoeForCausalLM` в
`hf_to_description` — по факту зонда реального чекпоинта
`Qwen/Qwen3-30B-A3B-Instruct-2507` (`~/models/qwen3-30b-a3b` на сервере,
61 ГБ, 16 BF16-шардов; read-only ssh). Бейзлайн ctest 52/52 → 52/52 (новая
секция 8c в `test_models_hf_description`, без новых исполняемых файлов).
НЕ пушено; чужие каталоги не тронуты (только `src/models`, `tests/models`).

## Зонд: факты реального чекпоинта (ssh read-only, safetensors-headers)

**config.json** (`model_type=qwen3_moe`, `architectures=["Qwen3MoeForCausalLM"]`):
`hidden_size=2048`, `num_hidden_layers=48`, `num_attention_heads=32`,
`num_key_value_heads=4`, `head_dim=128`, `vocab_size=151936`,
`rms_norm_eps=1e-06`, `rope_theta=1e7`, `num_experts=128`,
`num_experts_per_tok=8`, `moe_intermediate_size=768`,
`intermediate_size=6144` (dense-fallback), `norm_topk_prob=true`,
`decoder_sparse_step=1`, `mlp_only_layers=[]`, `tie_word_embeddings=false`.
**Ключи совпадают с Qwen2Moe** (тот же набор MoE-полей) — «честная
SchemaMismatch на отсутствующие» не потребовалась: отсутствующих ключей нет.

**Имена тензоров** (заголовок `model-00001-of-00016.safetensors`, 1062
тензора, уникальный суффикс — только `weight`):

- Верх: `model.embed_tokens.weight` BF16 `[151936,2048]`; в 16-м шарде —
  `lm_head.weight` BF16 `[151936,2048]` (tie=false) и `model.norm.weight`.
- Слой (каждый, включая слой 0): `input_layernorm`, `post_attention_layernorm`
  `[2048]`; `self_attn.{q,k,v,o}_proj` (q `[4096,2048]`, k/v `[512,2048]`,
  o `[2048,4096]` — dqq=4096 ≠ d_model=2048!); роутер **`mlp.gate.weight`**
  `[128,2048]`; эксперты `mlp.experts.{0..127}.{gate,up,down}_proj.weight`
  (gate/up `[768,2048]`, down `[2048,768]`).
- **Shared-expert: ПОДТВЕРЖДЕНО ОТСУТСТВИЕ** (0 тензоров с `shared` во всех
  просмотренных шардах) — однородный MoE без shared-expert.
- **Dense-слоёв НЕТ** (`dense mlp tensors: 0`; слой 0 — MoE) — все 48 слоёв
  MoE (`decoder_sparse_step=1`, `mlp_only_layers=[]`).
- **NEW (вне покрытых op'ов): `self_attn.q_norm.weight` / `self_attn.k_norm.weight`
  BF16 `[128]`** (=head_dim) в КАЖДОМ слое — per-head RMSNorm q/k (Qwen3-фича;
  в Qwen2/Qwen2Moe отсутствует).

## Реализация (`src/models/src/hf_description.cpp`)

1. **Arch в subset**: `Qwen3MoeForCausalLM` добавлен в семейство
   `moe_per_expert` (общая ветвь Qwen2Moe/Qwen1.5Moe/Olmoe); сообщение отказа
   о неизвестной архитектуре дополнено именем. Классификация слоёв —
   per-layer инфраструктура round 5/6 переиспользована без изменений
   (роутер `mlp.gate.weight` — суффикс-детекция уже покрывала Qwen1.5Moe/
   Olmoe-нейминг; dense-слои — llama-нейминг; отсутствие shared —
   цикл round-5 [R4] просто не создаёт групп: обратная совместимость,
   описание Qwen2Moe-фикстур побитово прежнее — тесты 52/52).
2. **Конфиг-маппинг**: `num_experts|num_local_experts`,
   `num_experts_per_tok` — как раньше; `head_dim` — как раньше (реальный
   чекпоинт задаёт явно 128); `moe_intermediate_size` — **перекрёстная
   сверка** с `d_ff`, выведенным из тензора первого MoE-слоя (несогласованность
   → честный SchemaMismatch, НЕ догадка; ключ отсутствует → прежнее
   поведение); `norm_topk_prob=false` → **честный SchemaMismatch** (схема
   фиксирует `"router":"softmax-topk"` с нормализацией вероятностей topk —
   как в Qwen2Moe/Qwen3Moe/Mixtral; отсутствие ключа — прежнее поведение).
3. **qk-norm → честный SchemaMismatch**: зонд вскрыл `self_attn.{q,k}_norm`
   — архитектурную особенность ВНЕ покрытых op'ов схемы: op attention
   исполняет проекции q/k БЕЗ qk-нормализации (см. `ops_runner.cpp`,
   `op_attention`), op rmsnorm нормирует только hidden. Генерировать группы
   нельзя: неизвестные группы ops_runner не потребляет — план исполнялся бы
   с неверными логитами (тихая фальшь, прямо запрещена ТЗ). Проверка — по
   ФАКТУ шардов (не по имени архитектуры) для всех поддержанных arch:
   любой слой с `self_attn.{q,k}_norm.weight` → SchemaMismatch с точной
   причиной (тензор, слой, непокрытая семантика, указание на поток core).
   **Следствие:** реальный Qwen3-30B-A3B в этом раунде честно НЕ
   конвертируется (всегда несёт qk-norm) — отказ стал ГРОМКИМ и точным
   вместо «архитектура не поддержана».
4. Обратная совместимость: Qwen2Moe-пути не изменены (52/52; отказ о
   неизвестной архитектуре в тестах не матчится по тексту — только код).

## Тесты (`tests/models/test_hf_description.cpp`, секция 8c)

- Фикстура `qwen3_moe_hf_tensors()` — тёзер-имена по факту зонда: роутер
  `mlp.gate.weight`, поэертельные `mlp.experts.{e}.{gate,up,down}_proj`,
  БЕЗ shared, `lm_head.weight` (tie=false); config-райтер
  `write_config_qwen3()` — ключи по факту зонда (`moe_intermediate_size`,
  `norm_topk_prob`, `head_dim`, `decoder_sparse_step=1`,
  `mlp_only_layers=[]`, dense-fallback `intermediate_size`).
- Happy path: описание генерируется — `class:"moe"`, `moe.n_experts=4`,
  `n_experts_active=2`, `expert_d_ff=8`; роутер-группа =
  `model.layers.{layer}.mlp.gate.weight` (зонд-нейминг); 43 группы после
  экспансии (embed+norm_final+head+2×kv+2×2 нормы+2×4 attn-матрицы+
  2 router+2×4×3 эксперта); **ни одной группы `ffn.shared`** (однородный
  MoE без shared — легитимно); ops = rmsnorm+attention+moe_gather_scatter
  (БЕЗ `ffn`); **roundtrip побитово** (слоты layout == прямому лоадеру,
  логиты == референсу плана, NVFP4-профиль).
- SchemaMismatch-кейсы: (а) qk-norm-фикстура (форма `[d_head]`, оба слоя —
  как в реальном чекпоинте) → SchemaMismatch, сообщение называет тензор
  и причину («op attention»); (б) `norm_topk_prob=false` → SchemaMismatch
  с ключом в сообщении; (в) `moe_intermediate_size=16` ≠ d_ff=8 шардов →
  SchemaMismatch с ключом в сообщении. Прежние 52 теста — без изменений.

## Ключевые решения

- **D4.** qk-norm — честный отказ, а не name-stub-группы: прецедент
  `ffn.shared.gate_scalar` ([R1] round 6) — это стыковая точка С
  согласованной core-частью; для qk-norm такой договорённости нет, а
  «группа без семантики в ops_runner» = тихо неверные логиты (ops_runner
  не валидирует непотреблённые группы — молча пропустит). Семантика
  per-head RMSNorm q/k (форма `[head_dim]`, нормализация каждой головы
  по `rms_norm_eps`) — предмет расширения схемы + `ops_runner` (поток core).
- **D5.** `norm_topk_prob` и `moe_intermediate_size` — честные проверки по
  факту конфига: ключ есть и противоречит схеме/шардам → громкий отказ;
  ключ отсутствует → прежнее поведение (старые фикстуры/чекпоинты не сломаны).
- **D6.** База ветки — локальный `master` `048500a` (merge models-7), т.к.
  `origin/master` в GitHub-remote отсутствует (fetch не вернул ref; в
  локальных remotes он «gone»). Содержимое базы проверено (`git log`).

## НЕ сделано / почему

- **Реальный серверный прогон Qwen3-30B-A3B** — ожидаемо даёт честный
  SchemaMismatch (qk-norm); пушить запрещено — команды для worker после
  merge (на сервере, в checkout'е): 
  `cmake --build build/cpu && ./build/cpu/src/models/stell_hf_description ~/models/qwen3-30b-a3b F16`
  → отказ с причиной «self_attn.q_norm.weight … op attention …». Для
  положительного сценария — фикстурный Qwen3Moe-класс (секция 8c) уже
  покрывает генерацию+roundtrip побитово.
- **Семантика qk-norm в ops_runner** — поток core (стыковая точка): расширить
  схему (параметр op attention или группы `l{layer}.attn.qk_norm`) и
  добавить per-head RMSNorm q/k; после этого в hf_to_description убрать
  отказ (заменить на группы) — одно место в коде, помечено комментарием.
- **dqq ≠ d_model в CPU-контуре** — наблюдение: у реального чекпоинта
  `n_heads*d_head=4096 ≠ d_model=2048` (и у Qwen2Moe-57B аналогично);
  `op_attention` (ops_runner, комментарий «fixture: n_heads*d_head ==
  d_model») пишет `joined[h*d_head+d]` в вектор длиной d_model — при
  dqq > d_model выход за границы. Схема описания такие модели ПРЕДСТАВЛЯЕТ
  корректно (d_head — отдельное поле; формы групп [dqq,dm] валидны);
  ограничение — исполнения, не описания: в зоне core. Не блокирует
  генерацию описаний Qwen3Moe-класса после внедрения qk-norm.

## Открытые вопросы

1. **qk-norm в схеме**: параметр op attention (например `qk_norm_eps` +
   роли групп) или отдельный op? Требуется решение core/planner (стыковая
   точка); форма тензоров `[head_dim]` на каждый слой зафиксирована зондом.
2. **dqq ≠ d_model**: подтвердить core-ограничение CPU-контура
   (`op_attention`, joined-проекция) и план (исправление в engine либо
   явный отказ лоадера/планёра на этом инварианте).
3. **`decoder_sparse_step=2`/`mlp_only_layers` реальных чекпоинтов** —
   покрываются per-layer классификацией round 5; реальных смешанных
   Qwen3Moe-чекпоинтов пока не известно (проверить на факте при появлении).

## Хеши коммитов (ветка `agent/models/8`, НЕ пушено)

- `2ab73a7` — models: Qwen3MoeForCausalLM в hf_to_description (зонд,
  реализация, честные отказы, тесты секции 8c, 52/52).
