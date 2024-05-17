import pm4py
import numpy as np
import pandas as pd
import random
import openpyxl
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.obj import PetriNet, Marking


# Получение отдельных трасс из журнала событий
#   Входные параметры:
#       path - абсолютный путь до файла в xes формате
#   Возвращаемое значение:
#       traces - список трасс
def read_log(path):
    log = pm4py.read_xes(path)
    traces = []
    for trace in np.unique(log['case:concept:name']):
        filter = pm4py.filter_event_attribute_values(log, 'case:concept:name', {trace}, level='event')
        traces.append(filter['concept:name'].to_numpy())
    return traces


# Создание матрицы отношений в рамках одного агента
#   Входные параметры:
#       rows - список событий, располагающихся в строках матрицы
#       columns - список событий, располагающихся в столбцах матрицы
#       traces - список трасс
#   Возвращаемое значение:
#       df_matrix - матрица отношений в виде дата фрейма
def create_matrix_one_agent(rows, columns, traces):
    matrix = np.full((len(rows), len(columns)), fill_value=" ", dtype=str)
    for trace in traces:
        for i in range(len(trace) - 1):
            for j in range(i + 1, len(trace)):
                row = rows.index(trace[i])
                column = columns.index(trace[j])
                if matrix[row, column] == " ":
                    matrix[row, column] = ">"
                    matrix[column, row] = "<"
                elif matrix[row, column] == "<":
                    matrix[row, column] = "+"
                    matrix[column, row] = "+"
                else:
                    continue
    ind = pd.Index(rows, name="rows")
    col = pd.Index(columns, name="columns")
    df_matrix = pd.DataFrame(data=matrix, index=ind, columns=col)

    return df_matrix


# Создание межагентной матрицы отношений
#   Входные параметры:
#       rows - список событий одного агента, располагающихся в строках матрицы
#       columns - список событий второго агента, располагающихся в столбцах матрицы
#       traces - список трасс
#   Возвращаемое значение:
#       df_matrix - матрица отношений в виде дата фрейма
def create_matrix_double_agent(rows, columns, traces):
    matrix = np.full((len(rows), len(columns)), fill_value=" ", dtype=str)
    for trace in traces:
        for i in range(len(trace) - 1):
            if (trace[i] not in rows) and (trace[i] not in columns):
                continue
            for j in range(i + 1, len(trace)):
                if ((trace[j] not in rows) and (trace[j] not in columns)) or (
                        (trace[i] in rows) and (trace[j] in rows)) or ((trace[i] in columns) and (trace[j] in columns)):
                    continue
                if (trace[i] in rows) and (trace[j] in columns):
                    row = rows.index(trace[i])
                    column = columns.index(trace[j])
                    if matrix[row, column] == " ":
                        matrix[row, column] = ">"
                    elif matrix[row, column] == "<":
                        matrix[row, column] = "+"
                    else:
                        continue
                if (trace[i] in columns) and (trace[j] in rows):
                    column = columns.index(trace[i])
                    row = rows.index(trace[j])
                    if matrix[row, column] == " ":
                        matrix[row, column] = "<"
                    elif matrix[row, column] == ">":
                        matrix[row, column] = "+"
                    else:
                        continue
    ind = pd.Index(rows, name="rows")
    col = pd.Index(columns, name="columns")
    df_matrix = pd.DataFrame(data=matrix, index=ind, columns=col)

    return df_matrix


# Получение кандидатов на точки асинхронного взаимодействия
#   Входные параметры:
#       matrix - межагентная матрица отношений
#   Возвращаемое значение:
#       interaction_points - массив точек асинхронного взаимодействия
def get_interaction_points(matrix):
    interaction_points = []
    rows = matrix.index
    columns = matrix.columns
    for i in range(len(rows)):
        for j in range(len(columns)):
            try:
                if matrix.iloc[i, j] == ">" and matrix.iloc[i + 1, j] != ">" and matrix.iloc[i, j - 1] != ">":
                    interaction_points.append([rows[i], columns[j]])
                if matrix.iloc[i, j] == "<" and matrix.iloc[i - 1, j] != "<" and matrix.iloc[i, j + 1] != "<":
                    interaction_points.append([columns[j], rows[i]])
            except:
                continue

    return interaction_points


# Рассчет граничных значений частоты событий для циклического взаимодействия
#   Входные параметры:
#       unique_a - уникальные события цикла первого агента
#       unique_b - уникальные события цикла второго агента
#   Возвращаемое значение:
#       mins - минимальные граничные значения
#       maxs - максимальные граничные значения
def cyclic_interaction(unique_a, unique_b, traces):
    mins = np.full((len(unique_a), len(unique_b)), fill_value=1000)
    maxs = np.full((len(unique_a), len(unique_b)), fill_value=-1000)
    for trace in traces:
        for i in range(len(unique_a)):
            for j in range(len(unique_b)):
                max_k = 0
                min_k = 0
                current = 0
                for t in trace:
                    if t == unique_a[i]:
                        current = current + 1
                    if t == unique_b[j]:
                        current = current - 1
                    max_k = np.max([current, max_k])
                    min_k = np.min([current, min_k])
                mins[i][j] = min(min_k, mins[i][j])
                maxs[i][j] = max(max_k, maxs[i][j])

    I = pd.Index(unique_a, name="rows")
    C = pd.Index(unique_b, name="columns")
    df_min = pd.DataFrame(data=mins, index=I, columns=C)
    print(df_min)
    I = pd.Index(unique_a, name="rows")
    C = pd.Index(unique_b, name="columns")
    df_max = pd.DataFrame(data=maxs, index=I, columns=C)
    print(df_max)

    return mins, maxs


# Разделение событий агента на ацикличную и цикличную части
#   Входные параметры:
#       unique_trans - уникальные события агента
#       traces - список трасс
#   Возвращаемое значение:
#       acyclic - список ацикличных событий
#       cyclic - список цикличных событий
def acyclic_cyclic_split(unique_trans, traces):
    acyclic = []
    cyclic = []
    for trans in unique_trans:
        for trace in traces:
            if trace.count(trans) > 1:
                cyclic.append(trans)
                break
        acyclic.append(trans)
    
    return acyclic, cyclic


# Получение уникального списка событий агента
#   Входные параметры:
#       traces - список трасс
#   Возвращаемое значение:
#       uniq_trans - список уникальных событий во входных трассах
def get_uniq_trans(traces):
    all_trans = []
    for trace in traces:
        all_trans += list(trace)
    uniq_trans = list(set(all_trans))
    return uniq_trans


# Получение пар событий, находящихся в отношении параллельности
#   Входные параметры:
#       matrix - матрица отношений
#   Возвращаемое значение:
#       result - массив пар событий, находящихся в отношении параллельности
def get_parallel_pairs(matrix):
    row, col = np.where(matrix.to_numpy() == "+")
    result = []
    for i in range(len(row)):
        if [matrix.columns[col[i]], matrix.index[row[i]]] not in result:
            result.append([matrix.index[row[i]], matrix.columns[col[i]]])
    return result


# Определение наборов и порядка записи событий в матрицы отношений
#   Входные параметры:
#       matrix - матрица отношений
#       traces - список трасс
#   Возвращаемое значение:
#       right_order - списки событий для записи в строки / столбцы матрицы отношений
def get_dim_order(matrix, traces):
    arr = [[x] for x in matrix.index]
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            if matrix.loc[row, col] in (">", "<", "+"):
                arr[i].append(col)

    parallel_pairs = get_parallel_pairs(matrix)

    right_order = []
    for i in range(len(arr)):
        if not arr[i]:
            continue
        for pair in parallel_pairs:
            if pair[0] in arr[i] and pair[1] in arr[i]:
                continue
        for trace in traces:
            proection = [x for x in trace if x in arr[i]]
            if proection:
                right_order.append(proection)
                break

    right_order = [list(x) for x in set(tuple(x) for x in right_order)]

    return right_order


# Процесс работы

# Абсолютные пути для журналов событий и проекций на агентов
path_full_log = "/Users/anastasiaserstugina/PycharmProjects/DiplomAlgo/Compositional process discovery_experiment data/IP-1/IP-1_initial_log.xes"
path_agent_log = [
    "/Users/anastasiaserstugina/PycharmProjects/DiplomAlgo/Compositional process discovery_experiment data/IP-1/IP-1_agent_1_log.xes",
    "/Users/anastasiaserstugina/PycharmProjects/DiplomAlgo/Compositional process discovery_experiment data/IP-1/IP-1_agent_2_log.xes"]

# Чтение журналов событий
traces_full = read_log(path_full_log)
traces_agents = []
for path in path_agent_log:
    traces_agents.append(read_log(path))

# Получение наборов уникальных событий агентов
uniq_0 = get_uniq_trans(traces_agents[0])
uniq_1 = get_uniq_trans(traces_agents[1])

# Создание матриц отношений в рамках каждого агента
matrix_0 = create_matrix_one_agent(uniq_0, uniq_0, traces_agents[0])
matrix_1 = create_matrix_one_agent(uniq_1, uniq_1, traces_agents[1])

# Определение наборов и порядка событий агентов для записи в межагентную матрицу
agent_0 = get_dim_order(matrix_0, traces_agents[0])
agent_1 = get_dim_order(matrix_1, traces_agents[1])

# Построение межагентной матрицы и получение точек взаимодействия
points = []
for group_0 in agent_0:
    for group_1 in agent_1:
        matrix = create_matrix_double_agent(group_0, group_1, traces_full)
        minimum = get_interaction_points(matrix)
        points.append(minimum)
points = [x for xs in points for x in xs]
points = [list(x) for x in set(tuple(x) for x in points)]
points.sort(key = lambda row: row[0])

print(points)

# Чтение модели независимых агентов
path_pnml = "/Users/anastasiaserstugina/PycharmProjects/DiplomAlgo/Compositional process discovery_experiment data/IP-1/IP-1_separate.pnml"
net, initial_marking, final_marking = pm4py.read_pnml(path_pnml)

# Добавление точек взаимодействия в модель
for ind, point in enumerate(points):
    place = PetriNet.Place(f"int_{ind}")
    net.places.add(place)
    for trans in net.transitions:
        if trans.label == point[0]:
            petri_utils.add_arc_from_to(trans, place, net)
            break
    for trans in net.transitions:
        if trans.label == point[1]:
            petri_utils.add_arc_from_to(place, trans, net)
            break

# Просмотр и запись полученной модели с добавленными каналами асинхронного взаимодействия
pm4py.view_petri_net(net, initial_marking, final_marking)
pm4py.write_pnml(net, initial_marking, final_marking, "/Users/anastasiaserstugina/PycharmProjects/DiplomAlgo/Compositional process discovery_experiment data/IP-1/IP1-points.pnml")
