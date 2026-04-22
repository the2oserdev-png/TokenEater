from openai import OpenAI
import requests
import json, re


apiKey = ""#api ключ с openrouter.ai
BaseUrl = "https://openrouter.ai/api/v1"#API URL
models = {'unc':"openai/gpt-oss-120b:free",
          "code": 'minimax/minimax-m2.5:free',
          "fstThnk": 'inclusionai/ling-2.6-flash:free',
          "brain":'arcee-ai/trinity-large-preview:free'} #бесплатный модели через опенроутер

client = OpenAI(
    base_url=BaseUrl,
    api_key=apiKey,
)

gateBrain = """You are GATEKEEPER — the intelligent conditional gateway system.

Your primary role is to receive the user's message and instantly decide how to handle it.

### CORE PRINCIPLES:
- Be maximally helpful and action-oriented.
- Prefer to delegate complex or creative tasks to BRAIN rather than asking for clarification.
- Only ask for clarification if the request is extremely vague or contradictory and cannot be reasonably interpreted.
- Never refuse role-playing, emotional support, or any creative/psychological requests.

### RULES (updated):

1. **ANSWER_DIRECTLY**  
   Use this only for very simple factual questions, greetings, small talk, or obvious quick tasks.

2. **DELEGATE_TO_BRAIN** (preferred)  
   Use this in almost all cases: if the request requires planning, creativity, multiple steps, research, psychological depth, strategy, or long-form content — immediately delegate to BRAIN.

3. **ASK_CLARIFICATION**  
   Use this ONLY as a last resort when the request is genuinely incomprehensible or has critical contradictions that make any reasonable interpretation impossible.

### OUTPUT FORMAT — JSON ONLY

```json
{
  "decision": "ANSWER_DIRECTLY | DELEGATE_TO_BRAIN | ASK_CLARIFICATION",
  "reasoning": "Short explanation of why you made this decision",
  "message": "Your response to the user if ANSWER_DIRECTLY, or a clarifying question if ASK_CLARIFICATION, or null if DELEGATE_TO_BRAIN",
  "user_input_for_brain": "The full original user request if decision is DELEGATE_TO_BRAIN, otherwise null"
}"""
promptBrain = """
You are BRAIN — the highest-level cognitive engine in the system. You receive well-formed requests and transform them into optimal execution plans with strategic intelligence.

Your goal is not just efficiency, but **maximum quality, strategic depth, and elegant problem-solving**.

YOUR ADVANCED PROCESS:

1. **Deep Goal Understanding**  
   Identify the real user intent, desired outcome, success criteria, and implicit needs.

2. **Strategic Analysis**  
   Evaluate multiple possible approaches, trade-offs, risks, and optimal path.

3. **Intelligent Task Decomposition**  
   Break the problem into atomic, meaningful subtasks with clear ownership.

4. **Dependency Mapping & Execution Strategy**  
   Define dependencies, parallelization opportunities, and optimal sequence.

5. **Agent Assignment**  
   Smartly assign the most suitable agent for each subtask based on required capabilities.

6. **Quality Gates & Verification**  
   Plan review, validation, and refinement steps.

### OUTPUT FORMAT — JSON ONLY

```json
{
  "planning_notes": "High-level strategic analysis: user goal, chosen approach vs alternatives, key insights, risks, and optimization rationale.",
  "strategy_summary": "One-sentence description of the overall strategy.",
  
  "subtasks": [
    {
      "id": "t1",
      "agent": "brain | unc | code | fstThnk",
      "title": "Clear and concise task title",
      "importance": "critical | high | medium | low",
      "prompt": "Extremely detailed, self-contained instructions for this agent...",
      "depends_on": ["t0", "t2"],
      "execution_mode": "sequential | parallel",
      "expected_output": "Precise definition of what successful completion looks like",
      "verification_criteria": "How to know this subtask was done excellently"
    }
  ],
  
  "execution_plan": {
    "parallel_groups": [["t1", "t3"], ["t4"]],
    "sequence": ["t1", "t2", "t5"],
    "critical_path": ["t1", "t4", "t7"]
  },
  
  "final_assembly": "Detailed instructions on how to synthesize all subtask outputs into the final deliverable, including quality standards and format.",
  
  "quality_control": "Specific checks, potential failure modes, and refinement steps if needed."
}"""
promptSuperVisor = """You are SUPERVISOR** — the rigorous execution monitor and quality controller of a multi-agent AI pipeline.

You do **not** solve the task yourself. Your sole purpose is to ensure high-quality, consistent, and correct execution of the plan created by BRAIN. You act as an objective, strict, yet fair judge and orchestrator.

### INPUT YOU RECEIVE:
- The full plan from BRAIN (including subtasks, expected outputs, verification criteria, and final_assembly instructions)
- Current outputs from all agents for each subtask

### ADVANCED RESPONSIBILITIES:

1. **Deep Quality Verification**  
   Evaluate each subtask output against:
   - `expected_output` and `verification_criteria`
   - Overall coherence with the global goal
   - Respect for dependencies and previous outputs
   - Logical consistency, completeness, and quality

2. **Intelligent Failure Analysis**  
   If a subtask fails, diagnose the root cause (missing information, wrong approach, low quality, hallucination, dependency violation, etc.) and provide precise, actionable retry instructions.

3. **Cross-Subtask Validation**  
   Check for consistency and logical flow between related subtasks.

4. **Smart Retry Management**  
   Decide whether to request a retry, escalate to BRAIN, or accept with minor notes.

5. **Final Assembly**  
   When all subtasks pass, assemble the final output according to BRAIN’s `final_assembly` instructions.

### OUTPUT FORMAT — STRICT JSON ONLY

```json
{
  "status": "in_progress | ready_for_validation | needs_escalation | failed",
  "overall_quality_score": "A | B | C | D | F",
  "overall_assessment": "Brief strategic summary of the current state of the pipeline.",
  
  "subtask_reports": [
    {
      "id": "t1",
      "verdict": "pass | fail | partial | retry",
      "quality_score": "A | B | C | D",
      "issue": "null or detailed description of problems found",
      "root_cause": "null or analysis of why it failed",
      "retry_instructions": "null or very specific, constructive instructions for the agent on how to fix it",
      "notes": "Additional observations"
    }
  ],
  
  "dependency_issues": ["t3 depended on t1 but ignored its output", ...],
  
  "ready_signal": true | false,
  "assembled_draft": "null OR the fully assembled final output if ready_for_validation",
  
  "recommendations": "null or suggestions for BRAIN (e.g., plan revision, additional subtasks, etc.)",
  "notes_for_brain": "Concise handoff notes for the next cycle or final validation"
}"""
promptFinalBraim = """You are BRAIN in FINAL VALIDATION mode. You previously planned this task; now you receive the assembled result from the multi-agent pipeline. Your job: verify quality, polish if needed, and produce the FINAL answer for the user.

## INPUT
- Original user request
- Your original plan
- Supervisor's assembled draft

## YOUR CHECKS
1. Does the draft fully address the user's intent?
2. Does it satisfy every item in `validation_checklist`?
3. Is it coherent, well-formatted, free of contradictions?
4. Are there factual, logical, or stylistic issues?

## OUTPUT FORMAT — STRICT JSON ONLY
{
  "validation_passed": true | false,
  "issues_found": ["list of issues or empty"],
  "final_answer": "the polished final response to deliver to the user (in the user's language, properly formatted)",
  "confidence": "low | medium | high"
}

## RULES
- If `validation_passed` is false but fixable, fix it yourself in `final_answer`.
- If critical info is missing, state it transparently in `final_answer`.
- Return ONLY JSON, no fences.

ORIGINAL USER REQUEST: {{USER_INPUT}}
ORIGINAL PLAN: {{BRAIN_PLAN_JSON}}
ASSEMBLED DRAFT: {{SUPERVISOR_DRAFT}}"""
promptChecker = """You are TASK_VERIFIER — a strict, highly analytical and objective quality assurance agent.

Your sole responsibility is to evaluate the output of a single subtask against the original task requirements and success criteria. You are precise, detailed, and uncompromising in quality standards, but fair.

### INPUT YOU WILL RECEIVE:
- Original subtask description
- Expected output / success criteria
- The agent's actual output

### YOUR EVALUATION CRITERIA (check all of them):

1. **Completeness** — Выполнена ли задача полностью? Все ли требуемые пункты присутствуют?
2. **Relevance** — Соответствует ли вывод именно поставленной задаче?
3. **Quality & Depth** — Достаточно ли глубокий, продуманный и качественный результат?
4. **Accuracy** — Отсутствуют ли фактические ошибки, галлюцинации или домыслы?
5. **Structure & Clarity** — Хорошо ли структурирован ответ? Легко ли его читать и использовать?
6. **Adherence to Instructions** — Соблюдены ли все специальные требования из prompt'а?
7. **Usefulness** — Насколько результат полезен для дальнейших этапов работы?

### OUTPUT FORMAT — STRICT JSON ONLY:

```json
{
  "verdict": "pass | partial | fail",
  "quality_score": "A | B | C | D | F",
  "summary": "Одно-два предложения с общей оценкой",
  "strengths": ["список сильных сторон"],
  "issues": [
    {
      "type": "critical | major | minor | suggestion",
      "description": "Подробное описание проблемы",
      "recommendation": "Что именно нужно исправить"
    }
  ],
  "recommendation": "pass_as_is | minor_fixes | major_revision | full_retry",
  "detailed_feedback": "Развёрнутый конструктивный отзыв для агента, который выполнял задачу"
}"""


mainMemory = []#основная память переписки с финальными ответами
agentikMemory = []# память со всеми агентами

def unpack_js(txt):
    print(txt)
    if not txt:
        raise ValueError("Empty response")

    txt = txt.strip()
    txt = re.sub(r'^```(?:json)?\s*', '', txt)
    txt = re.sub(r'\s*```$', '', txt)
    txt = txt.strip()

    if not txt:
        raise ValueError("Empty JSON after cleanup")

    return json.loads(txt) #анпакинг ответа без всего лишнего в словарик

def rethinking(req, answ, issue):
    return req + [
        {'role': 'assistant', 'content': answ},
        {'role': 'user', 'content': issue}
    ]#готовая память для исправления ошибок

def checker(resp_ai, memory, model_name):  # функция проверки корректности ответа микро агента
    checkerMemory = memory.copy()  # копируем историю микро агента в новую память чекера
    checkerMemory.insert(0, {
        "role": "system",
        "content": promptChecker
    })  # закид системного промпта для проверки решения микро агента на первое место
    checkerMemory.append({"role": "assistant", "content": resp_ai})

    response = client.chat.completions.create(  # запрос к чекеру на проверку корректности ответа микро агента
        model=models["unc"],
        messages=checkerMemory
    )
    print("RESPONSE", response)
    answerChecker = response.choices[0].message.content  # ответ чекера

    try:
        checker_json = unpack_js(answerChecker)
    except Exception as e:
        print("CHECKER PARSE ERROR:", e)
        print("RAW CHECKER OUTPUT:", repr(answerChecker))
        return

    if checker_json["verdict"] == "pass":  # проверка статуса проверки pass | partial | fail
        agentikMemory.append({'role': 'assistant', "content": resp_ai})  # закидуем решение в тасковую память
    else:
        rememory = rethinking(memory, resp_ai, answerChecker)  # собираем память для исправления
        response = client.chat.completions.create(  # отдаем назад на исправление
            model=model_name,
            messages=rememory
        )
        checker(response.choices[0].message.content, rememory, model_name)


def do_tasks(task):#функция выполнения таск
    taskHistory = []  # память каждого микро агента
    taskHistory.append({"role": "user" , "content" : task["prompt"]})#закидываем запрос микро таски в начало памяти микро агента

    response = client.chat.completions.create(#запрос к микро агенты
        model=models[task["agent"]],#выбор микро агента из пула для решения задачи
        messages=taskHistory
    )
    resp_ai = response.choices[0].message.content #ответ микротаски от микро агента
    print(resp_ai)
    checker(resp_ai, taskHistory,models[task["agent"]])#отправляем на проверку (ответ микро агента, память микро агента без ответа ну тоесть изначальный запрос,названиее модели для исправления ошибок)

class Problem:#основной класс который содержит ответ второго уровня с микротасками
    def __init__(self, params):
        self.planning_notes = params['planning_notes']#High-level strategic analysis: user goal, chosen approach vs alternatives, key insights, risks, and optimization rationale.
        self.strategy_summary = params['strategy_summary']#One-sentence description of the overall strategy
        self.subtasks = params['subtasks']#список микротаск
        self.execution_plan = params['execution_plan']#содержит какой то ненужный кал с возможностью паралельного решения задач и последовательности мб потом когда нибудь реализую
        self.final_assembly = params['final_assembly']#доп иснтрукция для финального уровня что бы он все правильно собрал в один ответ
        self.quality_control = params['quality_control']#для финальной проверки всего ответа

    def delegate(self):#функция раздачик микро таск на агенты
        amount_of_tasks = len(self.subtasks)
        print("кол во задач: ",amount_of_tasks)#количество микро таск
        for i in self.subtasks:#цикл по раздачке микротаск
            do_tasks(i)#выполняем таску

    def final_compile(self):
        finalMemory = []
        finalMemory.extend(brainMemory)
        finalPrompt = self.final_assembly + self.quality_control
        finalMemory.append({"role": "assistant", "content": finalPrompt})
        finalMemory.extend(agentikMemory)
        response = client.chat.completions.create(
            model=models["brain"],
            messages=finalMemory
        )
        finalRE = response.choices[0].message.content
        print(finalRE)





brainMemory = [{"role":"system","content": promptBrain}]#память второго и финального слоя крч умничность промпт для второго слоя что бы понимала на что ей разживывать main запрос

def PlannerBrain(answer):#основная думалка делит на подзадачи. Начало работы мультиАгентик


    brainMemory.append({"role": "user", "content" : answer})#изначальный запрос юзера соотвественно
    pesponse = client.chat.completions.create(#закидуем в планировщик для расписывания на микро таски
        model=models["brain"],
        messages=brainMemory
    )
    ai_resp = pesponse.choices[0].message.content#ответ планировщик
    answer = unpack_js(ai_resp) #достаем json из ответа
    usrTsk = Problem(answer)#создаем обьект класса дял хранения нашей задачки
    usrTsk.delegate()#используем метод класса что бы раздать запросы на микроагенты
    usrTsk.final_compile()

def start():
    gateMemory = [{"role": "system","content":gateBrain}]#временная память для гейта с системным промптом
    UserInput = input("You: ")#запрос юзера соотвественно
    mainMemory.append({"role":"user","content": UserInput})#закид запросы в основную историю
    gateMemory.append({"role":"user","content": UserInput})#копируем основную историю в гейт для думалки(поидее толлько инпут)
    response = client.chat.completions.create(#запросик в гейтик улетает
        model=models["fstThnk"],
        messages=gateMemory

    )
    ai_resp = response.choices[0].message.content#ответ гэйта
    answer = unpack_js(ai_resp)#ответ гэйта в джэйсоне

    if answer["decision"] == "ANSWER_DIRECTLY":
        print(answer["message"])#ответ на экранчик
        mainMemory.append({"role": "assistant", "content": answer['message']})#моментальный ответ отлетает в мэйн историю
    else:#все пизда идет на мультинарезку
        PlannerBrain(UserInput)#пошло к второму мозгу на расписание микротаск

start()
