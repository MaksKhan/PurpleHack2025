import sys
import json
import random
import math
from copy import deepcopy
from typing import Dict, List, Tuple
import heapq
import numpy as np

class GeneticScheduler:
    def __init__(self):
        self.hosts_info = {}       # {host: {"cpu": int, "ram": int}}
        self.allocation = {}       # {host: [vm1, vm2, ...]}
        self.host_usage = {}       # {host: {"cpu": int, "ram": int}}
        self.vm_specs = {}         # {vm: {"cpu": int, "ram": int}}
        self.current_vms = set()   # Текущий набор ВМ
        self.allocation_failures = set()  # Добавляем отслеживание неудачных размещений
        self.migration_history = set()
        self.host_usage_history = {}

        # Новые атрибуты для отслеживания простоя
        self.host_ever_used = {}  # {host: bool}
        self.host_idle_streak = {}  # {host: int}

    def _fitness(self, individual: Dict[str, List[str]]) -> float:
        total_score = 0.0
        migration_count = 0
        host_utilizations = []
    
        # Расчет миграций и утилизации
        for host, vms in individual.items():
            cpu_used = sum(self.vm_specs[vm]["cpu"] for vm in vms)
            ram_used = sum(self.vm_specs[vm]["ram"] for vm in vms)
            util = max(cpu_used/self.hosts_info[host]["cpu"], ram_used/self.hosts_info[host]["ram"])
            

            utilization_score = (-0.67459 + (42.38075 / (-2.5 * util + 5.96)) *
                             math.exp(-2 * (math.log(-2.5 * util + 2.96)) ** 2))

            
            # Дополнительный бонус за идеальную утилизацию
            if 0.78 <= util <= 0.82:
                utilization_score += 2
                
            total_score += utilization_score
                
            host_utilizations.append(util)
    
        # Штраф за дисбаланс утилизации между хостами
        balance_penalty = math.sqrt(np.var(host_utilizations)) * 10
        total_score -= balance_penalty
    
        # Штраф за миграции (учитываем историю)
        for vm in [vm for vms in individual.values() for vm in vms]:
            if vm in self.migration_history:
                migration_count += 1
        total_score -= (migration_count ** 2) * 0.3
    
        # Бонус за выключение хостов
        for host in self.hosts_info:
            if host not in individual or len(individual[host]) == 0:
                if self.host_usage_history[host]:
                    total_score += 800
                    
        return total_score

    def _generate_individual(self, new_vm: str = None) -> Dict[str, List[str]]:
        """
        Генерирует допустимого индивида:
        - Для новых ВМ пытается разместить их на случайных хостах.
        - Учитывает текущее распределение и ограничения.
        """
        individual = deepcopy(self.allocation)
        if new_vm:
            spec = self.vm_specs[new_vm]
            possible_hosts = [
                host for host in self.hosts_info
                if (self.host_usage[host]["cpu"] + spec["cpu"] <= self.hosts_info[host]["cpu"] and
                    self.host_usage[host]["ram"] + spec["ram"] <= self.hosts_info[host]["ram"])
            ]
            if possible_hosts:
                chosen_host = random.choice(possible_hosts)
                individual[chosen_host].append(new_vm)
        return individual

    def _mutate(self, individual: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Мутация: перемещает случайную ВМ на другой хост, если возможно.
        """
        vm_list = [vm for vms in individual.values() for vm in vms]
        if not vm_list:
            return individual
            
        vm = random.choice(vm_list)
        original_host = next(h for h, vms in individual.items() if vm in vms)
        spec = self.vm_specs[vm]
        
        individual[original_host].remove(vm)
        
        # Исправленная проверка ресурсов
        possible_hosts = [
            h for h in self.hosts_info
            if (sum(self.vm_specs[v]["cpu"] for v in individual[h]) + spec["cpu"] <= self.hosts_info[h]["cpu"])
            and (sum(self.vm_specs[v]["ram"] for v in individual[h]) + spec["ram"] <= self.hosts_info[h]["ram"])
        ]
        
        if possible_hosts:
            new_host = random.choice(possible_hosts)
            individual[new_host].append(vm)
        else:
            individual[original_host].append(vm)
            
        return individual

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Скрещивание: комбинирует распределения от двух родителей.
        """
        child = deepcopy(parent1)
        for host in parent2:
            if random.random() < 0.5:
                child[host] = parent2[host].copy()
        return child

    def _genetic_algorithm(self, new_vm: str = None) -> Dict[str, List[str]]:
        """
        Запускает генетический алгоритм для поиска оптимального распределения.
        """
        population_size = 80
        generations = 50
        population = [self._generate_individual(new_vm) for _ in range(population_size)]
        
        for _ in range(generations):
            # Оценка приспособленности
            fitness_scores = [self._fitness(ind) for ind in population]
            # Отбор лучших
            selected = heapq.nlargest(2, zip(fitness_scores, population), key=lambda x: x[0])
            # Скрещивание и мутация
            new_population = []
            for _ in range(population_size // 2):
                parent1 = random.choice(selected)[1]
                parent2 = random.choice(selected)[1]
                child = self._crossover(parent1, parent2)
                if random.random() < 0.2:
                    child = self._mutate(child)
                new_population.append(child)
            population = new_population
        
        best_individual = max(population, key=lambda x: self._fitness(x))
        return best_individual

    def process_round(self, round_input: str) -> str:
        round_input = json.loads(round_input)
        
        if not self.hosts_info:
            self.hosts_info = round_input.get("hosts", {})
            for host in self.hosts_info:
                self.allocation[host] = []
                self.host_usage[host] = {"cpu": 0, "ram": 0}
                self.host_usage_history[host] = False

        current_vms = round_input.get("virtual_machines", {})
        new_vms = set(current_vms.keys()) - self.current_vms
        removed_vms = self.current_vms - set(current_vms.keys())

        # Обработка удаления ВМ
        for vm in removed_vms:
            if vm in self.allocation_failures:
                self.allocation_failures.remove(vm)
            for host in self.allocation:
                if vm in self.allocation[host]:
                    self.allocation[host].remove(vm)
                    self.host_usage[host]["cpu"] -= self.vm_specs[vm]["cpu"]
                    self.host_usage[host]["ram"] -= self.vm_specs[vm]["ram"]
                    break
            if vm in self.vm_specs:  # Удаляем из спецификаций ПОСЛЕ обновления хост-usage
                del self.vm_specs[vm]
    
       # Обновляем данные
        self.vm_specs = current_vms.copy()
        self.current_vms = set(current_vms.keys())

        # Обработка добавления ВМ + перераспределение неудачных
        all_unallocated = list(new_vms) + list(self.allocation_failures)
        migrations = {}
        allocation_failures = []
        for vm in all_unallocated:
            best_individual = self._genetic_algorithm(vm)
            placed = False
            for host in best_individual:
                if vm in best_individual[host]:
                    # Проверяем миграции
                    original_host = next(
                        (h for h, vms in self.allocation.items() if vm in vms),
                        None
                    )
                    if original_host and host != original_host:
                        migrations[vm] = {"from": original_host, "to": host}
                        # Обновляем исходный хост
                        self.allocation[original_host].remove(vm)
                        self.host_usage[original_host]["cpu"] -= self.vm_specs[vm]["cpu"]
                        self.host_usage[original_host]["ram"] -= self.vm_specs[vm]["ram"]
                    # Обновляем новый хост
                    self.allocation[host].append(vm)
                    self.host_usage[host]["cpu"] += self.vm_specs[vm]["cpu"]
                    self.host_usage[host]["ram"] += self.vm_specs[vm]["ram"]
                    self.host_usage_history[host] = True
                    placed = True
                    break
            if not placed:
                allocation_failures.append(vm)
        # Сохраняем историю миграций
        self.migration_history.update(migrations.keys())
        # Обновляем список неудач с учетом новых и старых
        self.allocation_failures = set(allocation_failures)
        
        output = {
            "$schema": "resources/output.schema.json",
            "allocations": self.allocation,
            "allocation_failures": allocation_failures,
            "migrations": migrations
        }
        return json.dumps(output)

def main():
    scheduler = GeneticScheduler()
    for line in sys.stdin:
        response = scheduler.process_round(line.strip())
        print(response, flush=True)

if __name__ == "__main__":
    main()