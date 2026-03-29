import pygame
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from gui.renderer import Renderer
from algorithms import (
    generate_random_cities,
    create_solver,
    ALGORITHMS,
    calculate_route_distance,
)

# 覆寫 config.get_font 以提供支援中文的系統字體，解決 Pygame 中文顯示為方塊的問題
def get_chinese_font(size):
    if not pygame.font.get_init():
        pygame.font.init()
    
    # 常見的中文系統字體 (Windows: 微軟正黑體/微軟雅黑, Mac: 蘋方/黑體, Linux: 文泉驛)
    chinese_fonts = ['microsoftjhenghei', 'microsoftyahei', 'pingfang', 'stheiti', 'simhei', 'wqyzenhei']
    available_fonts = pygame.font.get_fonts()
    
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            return pygame.font.SysFont(font_name, size)
            
    return pygame.font.SysFont(None, size)  # 若找不到支援的中文字體，退回預設字體

config.get_font = get_chinese_font

class TSPVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode(config.WINDOW_DEFAULT_SIZE)
        pygame.display.set_caption("TSP 視覺化系統 - 旅行推銷員問題")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.running = True

        self.cities = {}
        self.current_route = []
        self.best_route = []
        self.best_distance = 0
        self.current_distance = 0
        self.chart_points = []
        self.new_route = None
        self.best_flash = 0
        self.hovered_city = None
        self.tooltip = None

        self.is_playing = False
        self.solver = None
        self.selected_algorithm = "爬山法"
        self.animation_speed = config.DEFAULT_ANIMATION_SPEED
        self.iterations_per_step = config.MAX_ITERATIONS_PER_STEP

        self.max_steps = 0
        self.current_step = 0

        self.buttons = []
        self.sliders = []
        self.algo_buttons = []
        self.init_ui()

    def init_ui(self):
        btn_width = config.CONTROL_PANEL_WIDTH - 40
        center_x = self.renderer.canvas_width + config.CONTROL_PANEL_WIDTH // 2

        self.buttons = [
            {"text": "隨機城市", "width": btn_width, "action": "random"},
            {"text": "清除全部", "width": btn_width, "action": "clear"},
            {
                "text": "開始",
                "width": btn_width // 2 - 5,
                "action": "start",
                "key": "play",
            },
            {
                "text": "暫停",
                "width": btn_width // 2 - 5,
                "action": "pause",
                "key": "pause",
            },
            {"text": "重置", "width": btn_width, "action": "reset"},
        ]

        for i, btn in enumerate(self.buttons):
            btn["x"] = center_x - btn_width // 2
            btn["y"] = 80 + i * 55
            btn["hover"] = False

        self.sliders = [
            {
                "label": "速度",
                "value": config.DEFAULT_ANIMATION_SPEED,
                "min": config.ANIMATION_SPEED_MIN,
                "max": config.ANIMATION_SPEED_MAX,
                "width": btn_width,
            },
            {
                "label": "步數限制",
                "value": 0,
                "min": 0,
                "max": 10000,
                "width": btn_width,
            },
        ]

        self.algo_buttons = []
        for algo in ALGORITHMS.keys():
            self.algo_buttons.append(
                {
                    "text": algo,
                    "width": btn_width,
                    "height": 40,
                    "action": "algorithm",
                    "algorithm": algo,
                    "selected": algo == self.selected_algorithm,
                    "hover": False,
                }
            )

    def get_canvas_rect(self):
        return pygame.Rect(
            0, 0, self.renderer.canvas_width, self.renderer.canvas_height
        )

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (
                        max(event.w, config.WINDOW_MIN_SIZE[0]),
                        max(event.h, config.WINDOW_MIN_SIZE[1]),
                    )
                )
                self.renderer = Renderer(self.screen)
                self.init_ui()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_left_click(mouse_pos)
                elif event.button == 3:
                    self.handle_right_click(mouse_pos)
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(mouse_pos)

        self.update_ui_state(mouse_pos)

    def handle_left_click(self, pos):
        canvas_rect = self.get_canvas_rect()
        panel_x = self.renderer.canvas_width

        for btn in self.buttons:
            if "rect" in btn and btn["rect"].collidepoint(pos):
                self.on_button_click(btn["action"])
                return

        for btn in self.algo_buttons:
            if "rect" in btn and btn["rect"].collidepoint(pos):
                self.on_algo_button_click(btn["algorithm"])
                return

        if canvas_rect.collidepoint(pos):
            self.add_city_at_pos(pos)
        else:
            slider_start_y = 415
            for i, slider in enumerate(self.sliders):
                x = panel_x + 20
                y = slider_start_y + i * 55
                width = slider.get("width", config.CONTROL_PANEL_WIDTH - 40)
                slider_rect = pygame.Rect(x, y + 22, width, 6)
                if slider_rect.collidepoint(pos):
                    self.update_slider_value(slider, pos[0] - x, width)
                    if slider["label"] == "步數限制":
                        self.max_steps = slider["value"]

    def handle_right_click(self, pos):
        canvas_rect = self.get_canvas_rect()
        if canvas_rect.collidepoint(pos):
            self.remove_city_at_pos(pos)

    def handle_mouse_motion(self, pos):
        self.hovered_city = None
        self.tooltip = None
        canvas_rect = self.get_canvas_rect()

        if canvas_rect.collidepoint(pos):
            for city_id, city_pos in self.cities.items():
                dx = pos[0] - city_pos[0]
                dy = pos[1] - city_pos[1]
                if math.sqrt(dx * dx + dy * dy) <= config.CITY_RADIUS:
                    self.hovered_city = city_id
                    self.tooltip = (f"城市 {city_id}: {city_pos}", pos)
                    break

    def add_city_at_pos(self, pos):
        for city_id, city_pos in self.cities.items():
            dx = pos[0] - city_pos[0]
            dy = pos[1] - city_pos[1]
            if math.sqrt(dx * dx + dy * dy) <= config.CITY_RADIUS * 2:
                return

        new_id = max([-1] + list(self.cities.keys())) + 1
        self.cities[new_id] = pos
        self.renumber_cities()
        self.update_routes()
        self.reset_solver()

    def remove_city_at_pos(self, pos):
        for city_id, city_pos in list(self.cities.items()):
            dx = pos[0] - city_pos[0]
            dy = pos[1] - city_pos[1]
            if math.sqrt(dx * dx + dy * dy) <= config.CITY_RADIUS + 5:
                del self.cities[city_id]
                self.renumber_cities()
                self.update_routes()
                self.reset_solver()
                break

    def renumber_cities(self):
        if not self.cities:
            return
        old_ids = sorted(self.cities.keys())
        new_cities = {}
        for new_id, old_id in enumerate(old_ids):
            new_cities[new_id] = self.cities[old_id]
        self.cities = new_cities

    def update_routes(self):
        if len(self.cities) >= 2:
            route = list(range(len(self.cities)))
            self.current_route = route
            self.best_route = route.copy()
            self.current_distance = calculate_route_distance(route, self.cities)
            self.best_distance = self.current_distance
        else:
            self.current_route = []
            self.best_route = []
            self.current_distance = 0
            self.best_distance = 0

    def reset_solver(self):
        self.solver = None
        self.chart_points = []
        self.new_route = None
        self.is_playing = False
        self.current_step = 0
        for btn in self.buttons:
            if btn.get("key") == "play":
                btn["active"] = False

    def on_button_click(self, action):
        if action == "random":
            self.generate_random_cities()
        elif action == "clear":
            self.clear_all()
        elif action == "start":
            self.start_algorithm()
        elif action == "pause":
            self.pause_algorithm()
        elif action == "reset":
            self.reset_algorithm()

    def generate_random_cities(self):
        self.cities = generate_random_cities(
            config.DEFAULT_CITY_COUNT,
            self.renderer.canvas_width - 20,
            self.renderer.canvas_height - 20,
            config.CITY_POSITIONS_MARGIN,
        )
        self.update_routes()
        self.reset_solver()
        self.update_algo_buttons()

    def clear_all(self):
        self.cities = {}
        self.current_route = []
        self.best_route = []
        self.chart_points = []
        self.new_route = None
        self.solver = None
        self.is_playing = False
        self.current_step = 0
        self.update_algo_buttons()
        for btn in self.buttons:
            if btn.get("key") == "play":
                btn["active"] = False

    def start_algorithm(self):
        if len(self.cities) < 2:
            return
        if self.solver is None:
            self.solver = create_solver(self.selected_algorithm, self.cities)
            self.chart_points = []
            self.current_step = 0
        self.is_playing = True
        for btn in self.buttons:
            if btn.get("key") == "play":
                btn["active"] = True
            elif btn.get("key") == "pause":
                btn["active"] = False

    def pause_algorithm(self):
        self.is_playing = False
        for btn in self.buttons:
            if btn.get("key") == "play":
                btn["active"] = False
            elif btn.get("key") == "pause":
                btn["active"] = False

    def reset_algorithm(self):
        if self.solver is not None and len(self.cities) >= 2:
            self.solver.reset()
            self.update_routes()
            self.chart_points = []
            self.new_route = None
            self.best_flash = 0
            self.current_step = 0
        self.is_playing = False
        for btn in self.buttons:
            if btn.get("key") == "play":
                btn["active"] = False

    def update_slider_value(self, slider, x, width):
        ratio = max(0, min(1, x / width))
        value = int(slider["min"] + ratio * (slider["max"] - slider["min"]))
        slider["value"] = value
        if slider["label"] == "速度":
            self.animation_speed = value
            self.iterations_per_step = max(1, value * 10)

    def update_ui_state(self, mouse_pos):
        for btn in self.buttons:
            if "rect" in btn:
                btn["hover"] = btn["rect"].collidepoint(mouse_pos)

        for btn in self.algo_buttons:
            if "rect" in btn:
                btn["hover"] = btn["rect"].collidepoint(mouse_pos)

    def update_algo_buttons(self):
        for btn in self.algo_buttons:
            btn["selected"] = btn["algorithm"] == self.selected_algorithm

    def on_algo_button_click(self, algo):
        self.selected_algorithm = algo
        self.update_algo_buttons()
        self.reset_solver()

    def update(self):
        if self.is_playing and self.solver is not None and len(self.cities) >= 2:
            if self.max_steps > 0 and self.current_step >= self.max_steps:
                self.is_playing = False
                for btn in self.buttons:
                    if btn.get("key") == "play":
                        btn["active"] = False
                return

            improved = self.solver.step(self.iterations_per_step)
            state = self.solver.get_state()
            self.current_step += 1

            self.current_route = state["current_route"]
            self.current_distance = state["current_distance"]
            self.best_route = state["best_route"]
            self.best_distance = state["best_distance"]
            self.new_route = state["new_route"]

            if state.get("is_new_best", False):
                self.best_flash = 1.0

            if self.best_flash > 0:
                self.best_flash -= 0.05

            self.chart_points.append(
                {
                    "distance": self.best_distance,
                    "iteration": state["iteration"],
                }
            )
            if len(self.chart_points) > config.CHART_MAX_POINTS:
                self.chart_points.pop(0)

            if state.get("is_complete", False):
                self.is_playing = False
                for btn in self.buttons:
                    if btn.get("key") == "play":
                        btn["active"] = False

    def draw(self):
        self.renderer.clear()

        canvas_rect = self.get_canvas_rect()
        pygame.draw.rect(self.screen, config.CANVAS_BG_COLOR, canvas_rect)

        if len(self.cities) >= 2:
            self.renderer.draw_all_routes(
                self.cities,
                self.current_route,
                self.best_route,
                self.new_route,
                max(0, self.best_flash),
            )

            for city_id, pos in self.cities.items():
                if self.hovered_city == city_id:
                    pygame.draw.circle(
                        self.screen,
                        config.CITY_HOVER_COLOR,
                        (int(pos[0]), int(pos[1])),
                        config.CITY_RADIUS + 3,
                        2,
                    )
        else:
            self.renderer.draw_info_text(
                "點擊畫布新增城市，或點擊「隨機城市」按鈕",
                (20, self.renderer.canvas_height // 2 - 10),
                20,
                config.TEXT_SECONDARY_COLOR,
            )

        if len(self.cities) >= 2:
            info_y = 10
            self.renderer.draw_info_text(f"城市數量: {len(self.cities)}", (10, info_y))
            info_y += 20
            self.renderer.draw_info_text(
                f"目前距離: {self.current_distance:.2f}", (10, info_y)
            )
            info_y += 20
            self.renderer.draw_info_text(
                f"最佳距離: {self.best_distance:.2f}",
                (10, info_y),
                16,
                config.BEST_ROUTE_COLOR,
            )

        chart_x = 0
        chart_y = self.renderer.canvas_height
        chart_width = self.renderer.canvas_width
        chart_height = config.CHART_HEIGHT
        self.renderer.draw_chart(
            self.chart_points, chart_x, chart_y, chart_width, chart_height
        )

        stats = self.get_stats()
        self.renderer.draw_control_panel(
            self.buttons, self.sliders, stats, self.algo_buttons
        )

        if self.tooltip:
            self.renderer.draw_tooltip(self.tooltip[0], self.tooltip[1])

        pygame.display.flip()

    def get_stats(self):
        step_text = "無限制" if self.max_steps == 0 else str(self.current_step)
        return {
            "城市數量": len(self.cities),
            "最佳距離": f"{self.best_distance:.2f}",
            "當前步數": step_text,
            "速度": self.animation_speed,
            "演算法": self.selected_algorithm,
        }

    def get_algo_chinese_name(self, algo):
        return algo

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    pygame.init()
    visualizer = TSPVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
