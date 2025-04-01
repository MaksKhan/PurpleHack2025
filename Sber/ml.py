import json
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import networkx as nx

class ProjectEnv(gym.Env):
    def __init__(self, project_data):
        super(ProjectEnv, self).__init__()
        
        # Инициализация данных проекта
        self.original_data = project_data
        self.tasks = self._parse_tasks_recursive(project_data['tasks']['rows'])  # Изменено
        self.dependencies = self._parse_dependencies(project_data['dependencies']['rows'])
        self.calendar = self._parse_calendar(project_data['calendars']['rows'][0]['intervals'])
        
        # Построение графа зависимостей
        self.task_graph = nx.DiGraph()
        for task in self.tasks:
            self.task_graph.add_node(task['id'])
        for from_task, to_task in self.dependencies:
            self.task_graph.add_edge(from_task, to_task)
            
        # Определение пространства действий и состояний
        self.action_space = spaces.Discrete(len(self.tasks) * 2)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(len(self.tasks) * 3 + 1,),
            dtype=np.float32  # Явное указание типа
        )
        self.reset()

    def _parse_calendar(self, calendar_data):
        """Парсинг календаря проекта"""
        work_days = set()
        for entry in calendar_data:
            if entry['isWorking']:
                if entry['startDate'] and entry['endDate']:
                    start = self._parse_date(entry['startDate'])
                    end = self._parse_date(entry['endDate'])
                    current = start
                    while current <= end:
                        work_days.add(current.date())
                        current += timedelta(days=1)
    
        return work_days

    def _is_work_day(self, date):
        """Проверка рабочего дня по календарю"""
        return date.date() in self.calendar

    def reset(self, seed=None):
        super().reset(seed=seed)
        # Восстановление исходного состояния проекта
        self.current_dates = {task['id']: (self._parse_date(task['startDate']), 
                                         self._parse_date(task['endDate'])) 
                            for task in self.tasks}
        self.project_start = min([start for start, _ in self.current_dates.values()])
        self.project_end = max([end for _, end in self.current_dates.values()])
        return self._get_state(), {}

    def step(self, action):
        # Преобразование действия в сдвиг задачи
        task_idx = action // 2
        direction = action % 2  # 0 - уменьшить, 1 - увеличить
        task_id = self.tasks[task_idx]['id']
        
        # Применение сдвига с проверкой ограничений
        new_start, new_end = self._shift_task(task_id, -1 if direction == 0 else 1)
        
        # Расчет награды
        new_project_end = max([end for _, end in self.current_dates.values()])
        reward = (self.project_end - new_project_end).days
        self.project_end = new_project_end
        
        # Проверка завершения
        terminated = reward <= 0  # Завершаем, если не удается улучшить
        truncated = False  # Не используем прерывание по времени
        
        return self._get_state(), reward, terminated, truncated, {}

    def _shift_task(self, task_id, delta_days):
        """Рекурсивный сдвиг задачи и возврат новых дат"""
        # Получаем текущие даты задачи
        start, end = self.current_dates[task_id]
        
        # Рассчитываем новые даты
        new_start = self._add_work_days(start, delta_days)
        new_end = self._add_work_days(new_start, (end - start).days)
        
        # Обновляем даты текущей задачи
        self.current_dates[task_id] = (new_start, new_end)
        
        # Рекурсивно сдвигаем дочерние задачи
        task = next(t for t in self.tasks if t['id'] == task_id)
        for child in task['children']:
            child_start, child_end = self._shift_task(child['id'], delta_days)
            new_start = min(new_start, child_start)
            new_end = max(new_end, child_end)
        
        # Обновляем зависимости
        for successor in self.task_graph.successors(task_id):
            succ_start, succ_end = self._shift_task(
                successor, 
                (new_start - start).days
            )
            new_end = max(new_end, succ_end)
        
        return new_start, new_end  # Явно возвращаем новые даты

    def _add_work_days(self, start_date, delta):
        # Логика добавления рабочих дней с учетом календаря
        current_date = start_date
        days_left = abs(delta)
        step = 1 if delta > 0 else -1
        
        while days_left > 0:
            current_date += timedelta(days=step)
            if self._is_work_day(current_date):
                days_left -= 1
        return current_date

    def _is_work_day(self, date):
        # Проверка по календарю проекта
        return date.weekday() < 5  # Упрощенная проверка (реальная логика должна использовать calendar)

    def _get_state(self):
        state = []
        total_duration = (self.project_end - self.project_start).days + 1e-8  # Избегаем деления на ноль
        
        for task in self.tasks:
            start, end = self.current_dates[task['id']]
            
            # Нормализация дат относительно всего проекта
            start_norm = (start - self.project_start).days / total_duration
            end_norm = (end - self.project_start).days / total_duration
            
            # Ограничение значений в диапазоне [0, 1]
            start_norm = np.clip(start_norm, 0.0, 1.0)
            end_norm = np.clip(end_norm, 0.0, 1.0)
            
            state.extend([
                start_norm,
                end_norm,
                1.0 if end > datetime.now() else 0.0
            ])
        
        # Нормализация общей длительности
        project_duration_norm = (self.project_end - self.project_start).days / 365.0
        state.append(np.clip(project_duration_norm, 0.0, 1.0))
        
        return np.array(state, dtype=np.float32)

    # Вспомогательные методы для парсинга данных
    def _parse_tasks_recursive(self, tasks, parent_id=None):
        """Рекурсивный парсинг задач с учетом вложенности"""
        parsed = []
        for task in tasks:
            # Добавляем текущую задачу
            parsed.append({
                'id': task['id'],
                'parent_id': parent_id,
                'startDate': task['startDate'],
                'endDate': task['endDate'],
                'duration': task['duration'],
                'children': []
            })
            
            # Рекурсивно обрабатываем детей
            if 'children' in task and len(task['children']) > 0:
                parsed[-1]['children'] = self._parse_tasks_recursive(
                    task['children'], 
                    parent_id=task['id']
                )
                parsed.extend(parsed[-1]['children'])
        
        return parsed

    def _parse_dependencies(self, deps):
        return [(dep['from'], dep['to']) for dep in deps]

    def _parse_date(self, date_str):
        return datetime.fromisoformat(date_str.replace('Z', ''))

def optimize_project(input_json):
    # Загрузка данных проекта
    project_data = json.loads(input_json)
    
    # Создание и обучение модели
    env = ProjectEnv(project_data)
    check_env(env)
    
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=5000)
    
    # Применение оптимальной политики
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
    
    # Рекурсивная функция для обновления дат во всей иерархии задач
    def update_task_dates(tasks):
        for task in tasks:
            # Обновляем текущую задачу
            if task['id'] in env.current_dates:
                new_start, new_end = env.current_dates[task['id']]
                task['startDate'] = new_start.isoformat()
                task['endDate'] = new_end.isoformat()
            
            # Рекурсивно обновляем дочерние задачи
            if 'children' in task and len(task['children']) > 0:
                update_task_dates(task['children'])
    
    # Начинаем обновление с корневых задач
    update_task_dates(project_data['tasks']['rows'])
    
    return json.dumps(project_data, ensure_ascii=False, indent=2)