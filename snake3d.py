import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random, sys, math
import numpy as np
import pygame.freetype  # For high-res text rendering

# -------------------------------
# Global Display Settings
# -------------------------------
display = (800, 600)
fullscreen = False

def set_display_mode(size, fs=False):
    flags = DOUBLEBUF | OPENGL | RESIZABLE
    if fs:
        flags |= FULLSCREEN
    return pygame.display.set_mode(size, flags)

def set_projection(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

# -------------------------------
# Initialization
# -------------------------------
pygame.init()
pygame.freetype.init()
pygame.mixer.init()

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

WIN = set_display_mode(display, fs=fullscreen)
pygame.display.set_caption("3D Snake")
set_projection(*display)

# -------------------------------
# Sound Setup
# -------------------------------
def generate_sound(frequency=440, duration=0.2, volume=0.25, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = volume * np.sin(2 * math.pi * frequency * t)
    waveform_int = np.int16(waveform * 32767)
    mixer_info = pygame.mixer.get_init()
    if mixer_info is not None and mixer_info[2] == 2:
        waveform_int = np.column_stack((waveform_int, waveform_int))
    return pygame.sndarray.make_sound(waveform_int)

sound_on = True
try:
    eat_sound = pygame.mixer.Sound("eat.wav")
    move_sound = pygame.mixer.Sound("move.wav")
    gameover_sound = pygame.mixer.Sound("gameover.wav")
except Exception as e:
    print("Wave files not found, using generated sounds.")
    eat_sound = generate_sound(330, 0.2, 0.25)
    move_sound = generate_sound(220, 0.1, 0.15)
    gameover_sound = generate_sound(110, 0.5, 0.25)

# -------------------------------
# OpenGL Shader Setup
# -------------------------------
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        print("Shader compile error:", glGetShaderInfoLog(shader))
        sys.exit()
    return shader

def create_shader_program():
    vertex_shader_source = """
    #version 120
    varying vec2 vTexCoord;
    varying vec3 vNormal;
    varying vec4 vColor;
    varying vec3 vPosition;
    void main() {
        vTexCoord = gl_MultiTexCoord0.xy;
        vNormal = normalize(gl_NormalMatrix * gl_Normal);
        vColor = gl_Color;
        vPosition = vec3(gl_ModelViewMatrix * gl_Vertex);
        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    }
    """
    fragment_shader_source = """
    #version 120
    varying vec2 vTexCoord;
    varying vec3 vNormal;
    varying vec4 vColor;
    varying vec3 vPosition;
    uniform sampler2D normalMap;
    uniform vec3 lightDir;
    uniform vec4 ambientColor;
    uniform vec4 diffuseColor;
    void main() {
        vec3 mapNormal = texture2D(normalMap, vTexCoord).rgb;
        mapNormal = normalize(mapNormal * 2.0 - 1.0);
        vec3 perturbedNormal = normalize(vNormal + mapNormal * 0.7);
        float diff = max(dot(perturbedNormal, normalize(-lightDir)), 0.0);
        vec4 litColor = ambientColor + diffuseColor * diff;
        gl_FragColor = vColor * litColor;
    }
    """
    vs = compile_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        print("Shader linking error:", glGetProgramInfoLog(program))
        sys.exit()
    return program

shader_program = create_shader_program()
u_lightDir = glGetUniformLocation(shader_program, "lightDir")
u_ambientColor = glGetUniformLocation(shader_program, "ambientColor")
u_diffuseColor = glGetUniformLocation(shader_program, "diffuseColor")
glUseProgram(shader_program)
glUniform3f(u_lightDir, 0.5, 1.0, 0.3)
glUniform4f(u_ambientColor, 0.4, 0.4, 0.4, 1.0)
glUniform4f(u_diffuseColor, 1.0, 1.0, 1.0, 1.0)
glUseProgram(0)

# -------------------------------
# Procedural Normal Map
# -------------------------------
def create_normal_map_texture(width=64, height=64):
    normal_map = np.zeros((height, width, 3), dtype=np.uint8)
    for j in range(height):
        for i in range(width):
            u = (i + 0.5) / width
            v = (j + 0.5) / height
            X = 2 * (u - 0.5)
            Y = 2 * (v - 0.5)
            r2 = X * X + Y * Y
            if r2 <= 1.0:
                Z = math.sqrt(1.0 - r2)
            else:
                norm = math.sqrt(r2)
                X /= norm
                Y /= norm
                Z = 0
            normal_map[j, i, 0] = int((X * 0.5 + 0.5) * 255)
            normal_map[j, i, 1] = int((Y * 0.5 + 0.5) * 255)
            normal_map[j, i, 2] = int((Z * 0.5 + 0.5) * 255)
    return normal_map

normal_map_data = create_normal_map_texture(64, 64)
normal_map_texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, normal_map_texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, normal_map_data)
glBindTexture(GL_TEXTURE_2D, 0)

# -------------------------------
# GAME VARIABLES
# -------------------------------
GRID_MIN = 0
GRID_MAX = 9
step_size = 1.0

snake = [(5,5,5), (4,5,5), (3,5,5)]
direction = (1,0,0)
pending_direction = None
game_state = "start"  # "start" = demo mode; "running" = manual

move_delay_ms = 600
last_move_time = pygame.time.get_ticks()
FPS = 60
clock = pygame.time.Clock()

stats_line = ""
last_stats_update = pygame.time.get_ticks()
stats_accum = 0.0
stats_frames = 0

# -------------------------------
# CAMERA SETTINGS
# -------------------------------
camera_angle = 0.0
camera_direction = 1
camera_speed = 0.3
max_camera_angle = 20.0
radius = 25.0
transition_active = False
transition_duration = 500  # ms
transition_start_time = 0
transition_initial_angle = 0.0

# -------------------------------
# Helper Functions
# -------------------------------
def positions_equal(pos1, pos2):
    return pos1[0] == pos2[0] and pos1[1] == pos2[1] and pos1[2] == pos2[2]

def is_valid(pos):
    x, y, z = pos
    if x < GRID_MIN or x > GRID_MAX or y < GRID_MIN or y > GRID_MAX or z < GRID_MIN or z > GRID_MAX:
        return False
    if any(positions_equal(pos, seg) for seg in snake):
        return False
    return True

def random_apple():
    while True:
        pos = (random.randint(GRID_MIN, GRID_MAX),
               random.randint(GRID_MIN, GRID_MAX),
               random.randint(GRID_MIN, GRID_MAX))
        if not any(positions_equal(pos, seg) for seg in snake):
            return pos

apple = random_apple()

def get_random_color(pos, base_color):
    return base_color

def auto_direction():
    head = snake[0]
    best_dir = None
    best_dist = float('inf')
    possible = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    for d in possible:
        if (direction[0]+d[0]==0 and direction[1]+d[1]==0 and direction[2]+d[2]==0):
            continue
        test_head = (head[0]+d[0], head[1]+d[1], head[2]+d[2])
        if is_valid(test_head):
            dist = math.sqrt((test_head[0]-apple[0])**2 +
                             (test_head[1]-apple[1])**2 +
                             (test_head[2]-apple[2])**2)
            if dist < best_dist:
                best_dist = dist
                best_dir = d
    return best_dir if best_dir is not None else direction

# -------------------------------
# Cube Drawing Functions
# -------------------------------
def draw_cube(x, y, z, size=0.9, color=(0,1,0), override_color=None):
    glPushMatrix()
    glTranslatef(x, y, z)
    half = size / 2.0
    if override_color is not None:
        if len(override_color) < 4:
            override_color = override_color + (1.0,)
        glColor4fv(override_color)
    else:
        glColor3fv(color)
    vertices = [
        (-half, -half, -half),
        ( half, -half, -half),
        ( half,  half, -half),
        (-half,  half, -half),
        (-half, -half,  half),
        ( half, -half,  half),
        ( half,  half,  half),
        (-half,  half,  half)
    ]
    glBegin(GL_QUADS)
    # Front face
    glNormal3f(0,0,-1)
    glTexCoord2f(0,0); glVertex3fv(vertices[0])
    glTexCoord2f(1,0); glVertex3fv(vertices[1])
    glTexCoord2f(1,1); glVertex3fv(vertices[2])
    glTexCoord2f(0,1); glVertex3fv(vertices[3])
    # Back face
    glNormal3f(0,0,1)
    glTexCoord2f(0,0); glVertex3fv(vertices[4])
    glTexCoord2f(1,0); glVertex3fv(vertices[5])
    glTexCoord2f(1,1); glVertex3fv(vertices[6])
    glTexCoord2f(0,1); glVertex3fv(vertices[7])
    # Left face
    glNormal3f(-1,0,0)
    glTexCoord2f(0,0); glVertex3fv(vertices[0])
    glTexCoord2f(1,0); glVertex3fv(vertices[4])
    glTexCoord2f(1,1); glVertex3fv(vertices[7])
    glTexCoord2f(0,1); glVertex3fv(vertices[3])
    # Right face
    glNormal3f(1,0,0)
    glTexCoord2f(0,0); glVertex3fv(vertices[1])
    glTexCoord2f(1,0); glVertex3fv(vertices[5])
    glTexCoord2f(1,1); glVertex3fv(vertices[6])
    glTexCoord2f(0,1); glVertex3fv(vertices[2])
    # Top face
    glNormal3f(0,1,0)
    glTexCoord2f(0,0); glVertex3fv(vertices[3])
    glTexCoord2f(1,0); glVertex3fv(vertices[2])
    glTexCoord2f(1,1); glVertex3fv(vertices[6])
    glTexCoord2f(0,1); glVertex3fv(vertices[7])
    # Bottom face
    glNormal3f(0,-1,0)
    glTexCoord2f(0,0); glVertex3fv(vertices[0])
    glTexCoord2f(1,0); glVertex3fv(vertices[1])
    glTexCoord2f(1,1); glVertex3fv(vertices[5])
    glTexCoord2f(0,1); glVertex3fv(vertices[4])
    glEnd()
    glPopMatrix()

def draw_regular_cube_outline(x, y, z, size, base_color):
    half = size / 2.0
    vertices = [
        (-half, -half, -half),
        ( half, -half, -half),
        ( half,  half, -half),
        (-half,  half, -half),
        (-half, -half,  half),
        ( half, -half,  half),
        ( half,  half,  half),
        (-half,  half,  half)
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    outline_color = (base_color[0]*0.1, base_color[1]*0.1, base_color[2]*0.1)
    glPushMatrix()
    glTranslatef(x, y, z)
    glColor3fv(outline_color)
    glLineWidth(2)
    glBegin(GL_LINES)
    for e in edges:
        glVertex3fv(vertices[e[0]])
        glVertex3fv(vertices[e[1]])
    glEnd()
    glLineWidth(1)
    glPopMatrix()

def draw_regular_cube(x, y, z, size=0.9, base_color=(0,1,0), outline=False):
    draw_cube(x, y, z, size=size, color=base_color)
    if outline:
        draw_regular_cube_outline(x, y, z, size, base_color)

def draw_food_cube(x, y, z, size=0.9, base_color=(1,0,0)):
    draw_regular_cube(x, y, z, size=size, base_color=base_color, outline=True)

# -------------------------------
# Text & Overlay Functions
# -------------------------------
def draw_text(x, y, text, font_size=16, color=(255,255,255)):
    font = pygame.freetype.SysFont("Arial", font_size)
    surface, _ = font.render(text, fgcolor=color)
    text_data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    gl_x = x
    gl_y = display[1] - y - height
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(gl_x, gl_y)
    glTexCoord2f(1,0); glVertex2f(gl_x+width, gl_y)
    glTexCoord2f(1,1); glVertex2f(gl_x+width, gl_y+height)
    glTexCoord2f(0,1); glVertex2f(gl_x, gl_y+height)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDeleteTextures(texture_id)

def draw_controls():
    controls = [
        "Controls:",
        "Arrow Keys: Move in X/Y",
        "W: Move In (Z-)",
        "S: Move Out (Z+)",
        "M: Toggle Sound",
        "Q or ESC: Quit"
    ]
    if game_state == "start":
        controls.append("Press any key to take control")
    margin = 10
    offset = 10
    for line in controls:
        draw_text(margin, offset, line, font_size=16, color=(255,255,255))
        offset += 20

def draw_stats():
    margin = 10
    draw_text(margin, display[1]-30, stats_line, font_size=16, color=(255,255,255))

def draw_game_over():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, display[0], 0, display[1])
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    center_x = display[0] // 2
    center_y = display[1] // 2
    draw_text(center_x-70, center_y-20, "GAME OVER", font_size=28, color=(255,0,0))
    pygame.display.flip()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def wait_for_restart():
    # Wait for 3 seconds while polling for events.
    start_wait = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_wait < 3000:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                # If Q or ESC is pressed, quit; otherwise, skip wait.
                if event.key in (K_q, K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                else:
                    reset_game()
                    return
        pygame.time.wait(10)
    reset_game()

def reset_game():
    global snake, direction, apple, last_move_time, pending_direction, game_state, move_delay_ms, camera_angle, transition_active
    snake = [(5,5,5), (4,5,5), (3,5,5)]
    direction = (1,0,0)
    pending_direction = None
    apple = random_apple()
    last_move_time = pygame.time.get_ticks()
    game_state = "start"  # Return to demo mode.
    move_delay_ms = 600
    transition_active = False
    camera_angle = 0.0

# -------------------------------
# Playfield (Grid Cube)
# -------------------------------
def draw_grid_cube():
    glColor3f(1,1,1)
    glBegin(GL_LINES)
    corners = [
        (GRID_MIN, GRID_MIN, GRID_MIN),
        (GRID_MAX, GRID_MIN, GRID_MIN),
        (GRID_MAX, GRID_MAX, GRID_MIN),
        (GRID_MIN, GRID_MAX, GRID_MIN),
        (GRID_MIN, GRID_MIN, GRID_MAX),
        (GRID_MAX, GRID_MIN, GRID_MAX),
        (GRID_MAX, GRID_MAX, GRID_MAX),
        (GRID_MIN, GRID_MAX, GRID_MAX)
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for edge in edges:
        for idx in edge:
            glVertex3fv(corners[idx])
    glEnd()

# -------------------------------
# Global Statistics
# -------------------------------
stats_line = ""
last_stats_update = pygame.time.get_ticks()
stats_accum = 0.0
stats_frames = 0

# -------------------------------
# Camera Settings
# -------------------------------
camera_angle = 0.0
camera_direction = 1
camera_speed = 0.3
max_camera_angle = 20.0
radius = 25.0
transition_active = False
transition_duration = 500  # ms
transition_start_time = 0
transition_initial_angle = 0.0

# -------------------------------
# MAIN GAME LOOP
# -------------------------------
running = True
while running:
    frame_start = pygame.time.get_ticks()
    
    # CAMERA UPDATE
    if game_state == "start":
        camera_angle = (camera_angle + camera_speed) % 360
    else:
        if transition_active:
            t = (pygame.time.get_ticks() - transition_start_time) / transition_duration
            if t >= 1.0:
                camera_angle = 0.0
                transition_active = False
                camera_direction = 1
            else:
                camera_angle = transition_initial_angle * (1 - t)
        else:
            camera_angle += camera_direction * camera_speed
            if camera_angle > max_camera_angle:
                camera_angle = max_camera_angle
                camera_direction = -1
            elif camera_angle < -max_camera_angle:
                camera_angle = -max_camera_angle
                camera_direction = 1

    # EVENT HANDLING
    for event in pygame.event.get():
        if event.type == VIDEORESIZE:
            display = event.size
            WIN = set_display_mode(display, fs=fullscreen)
            set_projection(*display)
        elif event.type == KEYDOWN:
            if event.key == K_F11:
                fullscreen = not fullscreen
                WIN = set_display_mode(display, fs=fullscreen)
                set_projection(*display)
            elif event.key == K_m:
                sound_on = not sound_on
            elif event.key in (K_q, K_ESCAPE):
                running = False
            else:
                if game_state == "start":
                    game_state = "running"
                    transition_active = True
                    transition_start_time = pygame.time.get_ticks()
                    transition_initial_angle = camera_angle
                new_direction = None
                if event.key == K_LEFT:
                    new_direction = (-1,0,0)
                elif event.key == K_RIGHT:
                    new_direction = (1,0,0)
                elif event.key == K_UP:
                    new_direction = (0,1,0)
                elif event.key == K_DOWN:
                    new_direction = (0,-1,0)
                elif event.key == K_w:
                    new_direction = (0,0,-1)
                elif event.key == K_s:
                    new_direction = (0,0,1)
                if new_direction is not None:
                    if not (direction[0] + new_direction[0] == 0 and 
                            direction[1] + new_direction[1] == 0 and 
                            direction[2] + new_direction[2] == 0):
                        pending_direction = new_direction

    # SNAKE MOVEMENT UPDATE
    if pygame.time.get_ticks() - last_move_time > move_delay_ms:
        last_move_time = pygame.time.get_ticks()
        if game_state == "start":
            direction = auto_direction()
        elif pending_direction is not None:
            direction = pending_direction
            pending_direction = None
        head = snake[0]
        new_head = (head[0] + direction[0]*step_size,
                    head[1] + direction[1]*step_size,
                    head[2] + direction[2]*step_size)
        if (new_head[0] < GRID_MIN or new_head[0] > GRID_MAX or 
            new_head[1] < GRID_MIN or new_head[1] > GRID_MAX or 
            new_head[2] < GRID_MIN or new_head[2] > GRID_MAX):
            if sound_on and gameover_sound:
                gameover_sound.play()
            draw_game_over()
            wait_for_restart()
            continue
        if any(positions_equal(new_head, seg) for seg in snake):
            if sound_on and gameover_sound:
                gameover_sound.play()
            draw_game_over()
            wait_for_restart()
            continue
        snake.insert(0, new_head)
        if sound_on and move_sound:
            move_sound.play()
        if positions_equal(new_head, apple):
            if sound_on and eat_sound:
                eat_sound.play()
            apple = random_apple()
            move_delay_ms = int(move_delay_ms * 0.95)
        else:
            snake.pop()

    # STATISTICS UPDATE (every 500 ms)
    stats_frames += 1
    stats_accum += clock.get_fps()
    now = pygame.time.get_ticks()
    if now - last_stats_update >= 500:
        avg_fps = stats_accum / stats_frames if stats_frames > 0 else 0
        stats_line = "FPS: {:.1f}   Render: {} ms   Speed: {} ms".format(
            avg_fps, pygame.time.get_ticks() - frame_start, move_delay_ms)
        last_stats_update = now
        stats_accum = 0.0
        stats_frames = 0

    # RENDER 3D SCENE
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    cam_x = radius * math.sin(math.radians(camera_angle))
    cam_z = radius * math.cos(math.radians(camera_angle))
    cam_y = 15
    gluLookAt(cam_x, cam_y, cam_z, 5, 5, 5, 0, 1, 0)
    
    glUseProgram(shader_program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, normal_map_texture)
    glUniform1i(glGetUniformLocation(shader_program, "normalMap"), 0)
    glUniform3f(u_lightDir, 0.5, 1.0, 0.3)
    glUniform4f(u_ambientColor, 0.4, 0.4, 0.4, 1.0)
    glUniform4f(u_diffuseColor, 1.0, 1.0, 1.0, 1.0)
    draw_grid_cube()
    # Draw food (red regular cube with outline).
    draw_food_cube(apple[0], apple[1], apple[2], size=0.9, base_color=(1,0,0))
    # Draw snake pieces (green regular cubes with outline).
    for seg in snake:
        draw_regular_cube(seg[0], seg[1], seg[2], size=0.9, base_color=(0,1,0), outline=True)
    glUseProgram(0)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    # RENDER OVERLAYS
    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, display[0], 0, display[1])
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    draw_controls()
    draw_stats()
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    
    pygame.display.flip()
    render_time_ms = pygame.time.get_ticks() - frame_start
    clock.tick(FPS)

pygame.quit()
