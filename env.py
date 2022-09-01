from pdb import set_trace as TT
import random

from einops import rearrange, repeat
import gym
import networkx as nx
import numpy as np
import pygame


class TileTypes:
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    ENEMY = 3


def gen_random_map(h, w):
    """Generate a random map of the given size."""
    # Generate terrain with some distribution of empty/wall tiles
    map_arr = np.random.choice(2, size=(h-2, w-2), p=[0.8, 0.2])
    # Create a border of wall around the map
    map_arr = np.pad(map_arr, 1, mode="constant", constant_values=TileTypes.WALL)
    # Spawn the player at a random position
    player_pos = np.array([random.randint(1, h-2), random.randint(1, w-2)], dtype=np.uint8)
    map_arr[player_pos[0], player_pos[1]] = TileTypes.PLAYER

    return map_arr, player_pos


def discrete_to_onehot(map_disc, n_chan):
    """Convert a discrete map to a onehot-encoded map."""
    return np.eye(n_chan)[map_disc].transpose(2, 0, 1)


class Env(gym.Env):
    h = w = 18
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    tile_size = 16
    _n_tiles = 4
    tile_colors = np.array([
        [255, 255, 255],  # Empty
        [0, 0, 0],  # Wall
        [0, 255, 0],  # Player
        [255, 0, 0],  # Enemy
    ])
    dummy_map = np.pad(np.zeros((h-2, w-2)), 1, mode='constant', constant_values=1)
    dummy_map[0,0] = dummy_map[0, w-1] = dummy_map[h-1, 0] = dummy_map[h-1, w-1] = 0
    border_coords = np.argwhere(dummy_map == 1)

    def __init__(self):
        self.enemies = []
        self.n_enemies_to_spawn = 1
        self.n_enemies = 0
        # Generate a discrete encoding of the map.
        self.map_disc, self.player_position = gen_random_map(h=self.h, w=self.w)
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)
        self.player_action_space = gym.spaces.Discrete(4)
        self.screen = None

        pass

    def step(self, action):
        next_pos = self.player_position + self.directions[action]
        if self.map_disc[tuple(next_pos)] == TileTypes.WALL:
            return self.map_onehot, 0, False, {}
        
        self.map_disc[tuple(self.player_position)] = TileTypes.EMPTY
        self.player_position = next_pos
        self.map_disc[tuple(self.player_position)] = TileTypes.PLAYER
        self._update_map_onehot()

        # TODO: Check if player is neighboring an anemy(s), and respond accordingly.

        if self.n_enemies_to_spawn > 0:
            self.n_enemies_to_spawn -= 1
            self.n_enemies += 1
            enemy = self.spawn_enemy()
            self.enemies.append(enemy)

        for enemy in self.enemies:
            # TODO: factor this out into generic movement code -SE
            old_enemy_pos = enemy.pos
            next_enemy_pos = enemy.update(self.map_onehot, self.player_position)
            self.map_disc[tuple(old_enemy_pos)] = TileTypes.EMPTY
            self.map_disc[tuple(next_enemy_pos)] = TileTypes.ENEMY
            self._update_map_onehot()

        return self.map_onehot, 0, False, {}

    def spawn_enemy(self):
        """Spawn an enemy at a random position on the border of the map."""
        n_border_tiles = 2 * (self.h + self.w) - 4
        enemy_pos = self.border_coords[random.randint(0, len(self.border_coords))]
        self.map_disc[tuple(enemy_pos)] = TileTypes.ENEMY
        self._update_map_onehot()
        enemy = Enemy(enemy_pos)
        return enemy

    def _update_map_onehot(self):
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)

    def reset(self):
        self.n_enemies_to_spawn = 1
        self.n_enemies = 0
        self.enemies = []
        self.map_disc, self.player_position = gen_random_map(h=self.h, w=self.w)
        self.map_onehot = discrete_to_onehot(self.map_disc, self._n_tiles)
        obs = self.map_onehot

    def render(self, mode='human'):
        tile_size = self.tile_size
        # self.rend_im = np.zeros_like(self.int_map)
        # Create an int map where the last tiles in `self.tiles` take priority.
        map_disc = self.map_disc
        self.rend_im = self.tile_colors[map_disc]
        self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        if mode == "human":
            if self.screen is None:
                pygame.init()
                # Set up the drawing window
                self.screen = pygame.display.set_mode([self.h*self.tile_size, self.w*self.tile_size])
            pygame_render_im(self.screen, self.rend_im)
            return
        else:
            raise NotImplementedError


def pygame_render_im(screen, img):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()


def main():
    env = Env()
    for _ in range(100):
        env.reset()
        env.render()
        for _ in range(100):
            action = env.player_action_space.sample()
            env.step(action)
            env.render()
        TT()


class Enemy():
    def __init__(self, pos):
        self.pos = pos
        self.path = None

    def update(self, map_onehot, player_pos):
        """Update the enemy's path to the player."""
        self.path_to_player = shortest_path(map_onehot, traversable_tile_idxs=[TileTypes.EMPTY], src_pos=self.pos, 
            trg_pos=player_pos)
        TT()
        self.pos = self.path_to_player[1]
        return self.pos


def id_to_xy(idx, width):
    return idx // width, idx % width

def xy_to_id(x, y, width):
    return x * width + y


def shortest_path(map_onehot, traversable_tile_idxs, src_pos, trg_pos):
    src, trg = None, None
    graph = nx.Graph()
    _, width, height = map_onehot.shape
    size = width * height
    nontraversable_edge_weight = size
    graph.add_nodes_from(range(size))
    edges = []
    src, trg = xy_to_id(*src_pos, width), xy_to_id(*trg_pos, width)
    for u in range(size):
        ux, uy = id_to_xy(u, width)
        edge_weight = 1
        if np.all(map_onehot[traversable_tile_idxs, ux, uy] != 1):
            edge_weight = nontraversable_edge_weight
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        # adj_feats = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        neighbs = [xy_to_id(x, y, width) for x, y in neighbs_xy]
        for v, (vx, vy) in zip(neighbs, neighbs_xy):
            if not 0 <= v < size or vx < 0 or vx >= width or vy < 0 or vy >= height:
                continue
            if np.all(map_onehot[traversable_tile_idxs, vx, vy] != 1):
                edge_weight = nontraversable_edge_weight
            graph.add_edge(u, v, weight=edge_weight)
            edges.append((u, v))
        edges.append((u, u))

    path = nx.shortest_path(graph, src, trg)
    path = np.array([id_to_xy(idx, width) for idx in path])

    return path


if __name__ == "__main__":
    main()
    
