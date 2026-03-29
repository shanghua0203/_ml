import pygame
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.width, self.height = screen.get_size()
        self.canvas_width = self.width - config.CONTROL_PANEL_WIDTH
        self.canvas_height = self.height - config.CHART_HEIGHT

    def clear(self):
        self.screen.fill(config.CANVAS_BG_COLOR)

    def draw_cities(
        self, cities, current_route, best_route, new_route=None, best_flash=0
    ):
        for city_id, pos in cities.items():
            x, y = pos
            color = config.CITY_COLOR
            if best_flash > 0 and city_id in best_route[:2]:
                color = (0, 255, 100)

            pygame.draw.circle(self.screen, color, (int(x), int(y)), config.CITY_RADIUS)
            font = config.get_font(16)
            text = font.render(str(city_id), True, config.CITY_TEXT_COLOR)
            text_rect = text.get_rect(center=(int(x), int(y)))
            self.screen.blit(text, text_rect)

    def draw_route(self, route, cities, color, width=2, dashed=False):
        if len(route) < 2:
            return
        points = [cities[city_id] for city_id in route]
        if dashed:
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                pygame.draw.line(self.screen, color, start, (mid_x, mid_y), width)
        else:
            for i in range(len(points) - 1):
                pygame.draw.line(self.screen, color, points[i], points[i + 1], width)

    def draw_return_line(self, route, cities, color, width=2):
        if len(route) < 2:
            return
        last_city = route[-1]
        first_city = route[0]
        pygame.draw.line(
            self.screen, color, cities[last_city], cities[first_city], width
        )

    def draw_all_routes(
        self, cities, current_route, best_route, new_route=None, best_flash=0
    ):
        if len(cities) < 2:
            return

        if new_route is not None:
            self.draw_route(new_route, cities, config.NEW_ROUTE_COLOR, 1, dashed=True)
            self.draw_return_line(new_route, cities, config.NEW_ROUTE_COLOR, 1)

        self.draw_route(current_route, cities, config.CURRENT_ROUTE_COLOR, 1)
        self.draw_return_line(current_route, cities, config.CURRENT_ROUTE_COLOR, 1)

        if best_flash > 0:
            flash_intensity = int(150 * best_flash)
            best_color = (0, 255, 100)
        else:
            best_color = config.BEST_ROUTE_COLOR
        self.draw_route(best_route, cities, best_color, 3)
        self.draw_return_line(best_route, cities, best_color, 3)

        self.draw_cities(cities, current_route, best_route, new_route, best_flash)

    def draw_control_panel(self, buttons, sliders, stats, algorithm_buttons):
        panel_x = self.canvas_width
        panel_rect = pygame.Rect(panel_x, 0, config.CONTROL_PANEL_WIDTH, self.height)
        pygame.draw.rect(self.screen, config.PANEL_BG_COLOR, panel_rect)

        y_offset = 15
        title_font = config.get_font(28)
        title = title_font.render("TSP 視覺化系統", True, config.TEXT_COLOR)
        title_rect = title.get_rect(
            centerx=panel_x + config.CONTROL_PANEL_WIDTH // 2, top=y_offset
        )
        self.screen.blit(title, title_rect)
        y_offset += 50

        pygame.draw.line(
            self.screen,
            config.PANEL_SECONDARY_BG,
            (panel_x + 20, y_offset),
            (panel_x + config.CONTROL_PANEL_WIDTH - 20, y_offset),
            2,
        )
        y_offset += 15

        for btn in buttons:
            self.draw_button(btn, y_offset)
            y_offset += btn.get("height", 45) + 10

        y_offset += 10

        for slider in sliders:
            self.draw_slider(slider, y_offset)
            y_offset += 55

        y_offset += 5
        pygame.draw.line(
            self.screen,
            config.PANEL_SECONDARY_BG,
            (panel_x + 20, y_offset),
            (panel_x + config.CONTROL_PANEL_WIDTH - 20, y_offset),
            2,
        )
        y_offset += 15

        algo_label = config.get_font(18).render("選擇演算法:", True, config.TEXT_COLOR)
        self.screen.blit(algo_label, (panel_x + 20, y_offset))
        y_offset += 28

        for btn in algorithm_buttons:
            btn["y"] = y_offset
            self.draw_button(btn, y_offset)
            y_offset += btn.get("height", 40) + 5

        y_offset += 10
        pygame.draw.line(
            self.screen,
            config.PANEL_SECONDARY_BG,
            (panel_x + 20, y_offset),
            (panel_x + config.CONTROL_PANEL_WIDTH - 20, y_offset),
            2,
        )
        y_offset += 15

        stats_label = config.get_font(18).render("統計資訊:", True, config.TEXT_COLOR)
        self.screen.blit(stats_label, (panel_x + 20, y_offset))
        y_offset += 28

        for key, value in stats.items():
            text = config.get_font(16).render(
                f"{key}: {value}", True, config.TEXT_SECONDARY_COLOR
            )
            self.screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 22

    def draw_button(self, button, y_offset=None):
        x = button.get(
            "x",
            self.canvas_width + config.CONTROL_PANEL_WIDTH // 2 - button["width"] // 2,
        )
        if y_offset is None:
            y_offset = button.get("y", 100)
        rect = pygame.Rect(x, y_offset, button["width"], button.get("height", 45))
        button["rect"] = rect

        is_hover = button.get("hover", False)
        is_active = button.get("active", False)

        if is_active:
            color = config.BUTTON_ACTIVE_COLOR
        elif is_hover:
            color = config.BUTTON_HOVER_COLOR
        else:
            color = config.BUTTON_BG_COLOR

        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=8)

        text = config.get_font(button.get("font_size", 20)).render(
            button["text"], True, config.TEXT_COLOR
        )
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_slider(self, slider, y_offset):
        x = slider.get("x", self.canvas_width + 20)
        width = slider.get("width", config.CONTROL_PANEL_WIDTH - 40)
        height = 25

        label = config.get_font(16).render(slider["label"], True, config.TEXT_COLOR)
        self.screen.blit(label, (x, y_offset))

        value_text = config.get_font(16).render(
            str(slider["value"]), True, config.TEXT_SECONDARY_COLOR
        )
        self.screen.blit(value_text, (x + width - 30, y_offset))

        track_rect = pygame.Rect(x, y_offset + 22, width, 6)
        pygame.draw.rect(
            self.screen, config.PANEL_SECONDARY_BG, track_rect, border_radius=3
        )

        ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
        knob_x = x + int(ratio * width)
        knob_rect = pygame.Rect(knob_x - 8, y_offset + 16, 16, height - 8)
        pygame.draw.rect(
            self.screen, config.BUTTON_BG_COLOR, knob_rect, border_radius=4
        )
        pygame.draw.rect(self.screen, (255, 255, 255), knob_rect, 1, border_radius=4)

        slider["rect"] = track_rect

    def draw_chart(self, points, chart_x, chart_y, chart_width, chart_height):
        chart_rect = pygame.Rect(chart_x, chart_y, chart_width, chart_height)
        pygame.draw.rect(self.screen, config.CHART_BG_COLOR, chart_rect)

        padding = 10
        inner_x = chart_x + padding
        inner_y = chart_y + padding
        inner_w = chart_width - padding * 2
        inner_h = chart_height - padding * 2

        pygame.draw.rect(
            self.screen,
            config.CHART_GRID_COLOR,
            (inner_x, inner_y, inner_w, inner_h),
            1,
        )

        if len(points) < 2:
            return

        min_dist = min(p["distance"] for p in points)
        max_dist = max(p["distance"] for p in points)
        if max_dist - min_dist < 0.001:
            max_dist = min_dist + 1

        x_step = inner_w / max(len(points) - 1, 1)

        for i in range(len(points) - 1):
            x1 = inner_x + i * x_step
            x2 = inner_x + (i + 1) * x_step
            y1 = (
                inner_y
                + inner_h
                - ((points[i]["distance"] - min_dist) / (max_dist - min_dist)) * inner_h
            )
            y2 = (
                inner_y
                + inner_h
                - ((points[i + 1]["distance"] - min_dist) / (max_dist - min_dist))
                * inner_h
            )
            pygame.draw.line(
                self.screen, config.CHART_LINE_COLOR, (x1, y1), (x2, y2), 2
            )

        if points:
            last_point = points[-1]
            x = inner_x + (len(points) - 1) * x_step
            y = (
                inner_y
                + inner_h
                - ((last_point["distance"] - min_dist) / (max_dist - min_dist))
                * inner_h
            )
            pygame.draw.circle(
                self.screen, config.CHART_POINT_COLOR, (int(x), int(y)), 4
            )

        label_font = config.get_font(12)
        dist_label = label_font.render(
            f"最佳: {min_dist:.2f}", True, config.TEXT_SECONDARY_COLOR
        )
        self.screen.blit(dist_label, (inner_x, chart_y + 2))

        iter_label = label_font.render(
            f"迭代: {points[-1]['iteration']}", True, config.TEXT_SECONDARY_COLOR
        )
        self.screen.blit(iter_label, (inner_x + inner_w - 60, chart_y + 2))

    def draw_info_text(self, text, pos, size=16, color=None):
        if color is None:
            color = config.TEXT_SECONDARY_COLOR
        font = config.get_font(size)
        surface = font.render(text, True, color)
        self.screen.blit(surface, pos)

    def draw_tooltip(self, text, pos):
        font = config.get_font(14)
        surface = font.render(text, True, (0, 0, 0))
        rect = surface.get_rect(center=pos)
        bg_rect = rect.inflate(10, 6)
        pygame.draw.rect(self.screen, (255, 255, 200), bg_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
        self.screen.blit(surface, rect)
