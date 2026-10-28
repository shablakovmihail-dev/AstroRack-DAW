#"""Стек Undo/Redo для состояний параметров обработки."""


class HistoryManager:
    """Хранит снимки состояния с ограничением глубины и указателем текущей позиции."""

    def __init__(self, max_depth: int = 50):
        self._max_depth = max(1, int(max_depth))
        self._stack = []
        self._index = -1

    def push(self, state, action_name: str = ""):
        if self._index < len(self._stack) - 1:
            self._stack = self._stack[: self._index + 1]
        if isinstance(state, dict) and action_name and '_action_name' not in state:
            state['_action_name'] = action_name
        self._stack.append(state)
        if len(self._stack) > self._max_depth:
            overflow = len(self._stack) - self._max_depth
            self._stack = self._stack[overflow:]
        self._index = len(self._stack) - 1

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._stack) - 1

    def undo(self):
        if not self.can_undo():
            return None
        self._index -= 1
        return self._stack[self._index]

    def redo(self):
        if not self.can_redo():
            return None
        self._index += 1
        return self._stack[self._index]

    def undo_action_name(self) -> str:
        if self.can_undo():
            state = self._stack[self._index]
            if isinstance(state, dict):
                return state.get('_action_name', 'Action')
        return ""

    def redo_action_name(self) -> str:
        if self.can_redo():
            state = self._stack[self._index + 1]
            if isinstance(state, dict):
                return state.get('_action_name', 'Action')
        return ""

    def clear(self):
        self._stack = []
        self._index = -1
