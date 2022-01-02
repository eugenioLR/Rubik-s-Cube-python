import traceback
from .video_input import WebcamVideoStream
from .image_processing_utils import *
from . import *

close_flag = False

def on_close(event):
    global close_flag
    close_flag = True

color_names = {0:"white", 1:"red", 2:"blue", 3:"orange", 4:"green", 5:"yellow", -2: "gray", -1:"purple?"}

def face_equal_norot(face1, face2):
    """
    We check if 2 faces are equal independently of their orientation
    """
    are_equal = False
    i = 0
    while i < 4 and not are_equal:
        are_equal = are_equal or (face1 == np.rot90(face2, i)).all()
        i += 1
    return are_equal


class Cube_calibrator:
    def __init__(self, config):
        # Set configuration from the config file
        self.fps = config["fps"]
        self.solver_type = config["solver"]
        self.segm_method = config["segm_method"]
        self.confirm_time = config["face_confirmation_time"]
        self.tick_time = config["tick_showing_time"]
        self.on_phone = config["on_phone"]
        self.debug = config["debug"]

        # Internal parameters
        self.cam = WebcamVideoStream(fps=self.fps)
        self.last_face = -1
        self.calibrated = False
        self.solution_found = False
        self.finished = False
        self.cube = Cube(3)
        self.cube_next = None
        self.solver = None

    def arrange_cube(self, faces, colors):
        """
        With the 6 faces of the cube and knowing that they are arranged in a certain order
        we put them in the correct position and orientation.

        We return a normalized cube (that has the top face as white, left face as red and front face as blue) 
        """

        opposites = {0:5,1:3,2:4,3:1,5:0}

        faces[0] = np.rot90(faces[0],3)
        faces[5] = np.rot90(faces[5],2)

        return Cube(3, faces).normalize()

    def get_shape_coords(self, shape_type, scale, offset, progress, rotation):
        """
        Gets the coordinates to draw some basic shapes with plots

        The shapes implemented are:
            - Circunference both complete and incomplete
            - Check mark or Tick
            - Straight arrow
            - Circle with arrow
            - Flat line
        """
        if shape_type == 'circle':
            rad_rot = np.pi*rotation/180
            max_theta = (2*np.pi)*progress
            theta = np.linspace(0, max_theta, 100)
            theta = theta + rad_rot
            indicator_x = scale*np.cos(theta)
            indicator_y = scale*np.sin(theta)

        elif shape_type == 'tick':
            indicator_x = scale*np.array([-0.5,0,0.75])
            indicator_y = scale*np.array([0,0.75,-0.75])

        elif shape_type == 'arrow':
            indicator_x = scale*np.array([-1,1,0.25,1,0.25])
            indicator_y = scale*np.array([0, 0, 0.5,0,-0.5])
            if rotation == 90:
                indicator_x, indicator_y = indicator_y, -indicator_x
            elif rotation == 180:
                indicator_x = -indicator_x
            elif rotation == 270:
                indicator_x, indicator_y = indicator_y, indicator_x

        elif shape_type == 'arrow_circle':
            theta = np.linspace(0, 3*np.pi/2, 100)
            indicator_x = scale*np.hstack([np.cos(theta), [0.2, -0.2, 0.2, -0.2, 0.2]])
            indicator_y = scale*np.hstack([np.sin(theta), [-1, -0.5, -1, -1.5, -1]])
            if rotation == 180:
                indicator_x = -indicator_x

        elif shape_type == 'dash':
            indicator_x = scale*np.array([-1,1])
            indicator_y = scale*np.array([0, 0])

        indicator_x = indicator_x + offset[1]
        indicator_y = indicator_y + offset[0]

        return indicator_x, indicator_y

    def main(self, phone=True):
        """
        Executes the real time program, split in 3 phases:
            -Calibration: the cube is calibrated by pointing the camera to it's
                6 faces in a determined order indicated to the user

            -Solving: a solver thread is launched and a progress circle is shown

            -Showing solution: the solution will be shown as well as the arrow
                indicating which move to make each step

        """
        self.cam.start()
        solver_thread = None
        try:
            ## Initialize plot

            # Make plots update
            plt.ion()

            # Create figure
            fig = plt.figure()

            # Make the closing of the window finish the program
            fig.canvas.mpl_connect('close_event', on_close)

            # Create a subplot to draw to
            ax = plt.subplot()

            # Initialize the subplot by drawing a picture and a line
            path = str(Path(__file__).resolve().parent) + "/"
            I_rgb = cv2.imread(path+"rubiks_cube_photo.png")
            im = ax.imshow(np.zeros(I_rgb.shape))
            indic, = ax.plot([0,0],[0,0], linewidth=10)
            plt.show()

            # Set indicator offset
            if phone:
                indicator_offset = [I_rgb.shape[0]*(7/8), I_rgb.shape[1]/2]
            else:
                indicator_offset = [I_rgb.shape[0]/2, I_rgb.shape[1]*(7/8)]

            # Set some parameters up
            frame_time = 1/self.fps
            start = time.time()
            face_confirm_timer = 0
            tick_timer = 0
            solution_timer = 0
            prev_face = np.zeros([1,1])
            colors_checked = []
            faces = []
            solution = []
            alpha = 1
            warned = False

            # Main loop
            while not self.finished and not close_flag:
                # Measure time for FPS control
                frame_start = time.time()

                # clear last frame's data
                ax.clear()


                ## Image Aquisition
                frame_original = self.cam.read()

                # The original image will be in BGR, we transform it to RGB
                frame_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)

                if self.on_phone:
                    frame_original = cv2.rotate(frame_original, cv2.ROTATE_90_CLOCKWISE)


                ## Pre-processing
                frame_hsv = cv2.cvtColor(frame_original, cv2.COLOR_RGB2HSV)


                frame_hsv = cv2.medianBlur(frame_hsv, 9)


                ## Extraction of characteristics

                # Get binarized image, various implementations, "binarize" works best
                if self.segm_method == "binarize":
                    frame_bw = binarize(frame_hsv)
                elif self.segm_method == "borders":
                    frame_bw = filled_borders(frame_hsv)
                else:
                    frame_bw = binarize(frame_hsv)
                    if not warned:
                        print(f"WARNING: \"{self.segm_method}\" is not a valid segmentation method, we default to \"binarize\"")
                        warned = True

                # Find the features to analyze
                contours, positions = find_contours(frame_bw, debug = False)


                ## Description
                face, face_positions, stiker_size = get_ordered_colors(frame_hsv, contours, debug = self.debug)

                ## User interaction
                # Initialize indicator data
                indic_size = 40
                indicator_x = []
                indicator_y = []
                indicator_color = "w"
                linestyle = "solid"

                if not self.calibrated:
                    ## Phase 1: Obtain the colors of the faces

                    plt.title("Cube calibration")
                    if face.ndim == 2 and (face == prev_face).all() and face[1,1] not in colors_checked:
                        # We are checking a new face

                        # Reset timer for the checkmark
                        tick_timer = time.time()

                        # Get progress circle to draw later
                        indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, (time.time() - face_confirm_timer)/self.confirm_time, 0)
                        indicator_color = "orange"

                        # When the timer finishes add the face to out list of faces
                        if time.time() - face_confirm_timer > self.confirm_time:
                            colors_checked.append(face[1,1])
                            faces.append(face)
                            self.last_face = face[1,1]
                            if len(faces) == 6:
                                self.calibrated = True

                    elif face.ndim == 2 and (face == prev_face).all() and face[1,1] in colors_checked and time.time() - tick_timer < self.tick_time:
                        # The face we are seeing is already stored

                        # Reset timer for face confirmation
                        face_confirm_timer = time.time()

                        # Get green check mark to draw later
                        indicator_x, indicator_y = self.get_shape_coords("tick", indic_size, indicator_offset, 0, 0)
                        indicator_color = "green"
                    else:
                        # Reset timer for face confirmation
                        face_confirm_timer = time.time()

                        # Update the face
                        prev_face = face

                        # Show user the direction in which they have to rotate the cube
                        if len(colors_checked) == 0:
                            indicator_x, indicator_y = self.get_shape_coords("dash", indic_size, indicator_offset, 0, 0)
                        elif len(colors_checked) in [1,5]:
                            indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, indicator_offset, 0, 270)
                        elif len(colors_checked) in [2,3,4]:
                            indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, indicator_offset, 0, 0)
                        indicator_color = "w"


                    # Show color text
                    if face_positions is not None:
                        face_list = face.flatten()
                        for i in range(face_positions.shape[1]):
                            # Calculate text size, linear equation with points (4000,10),(900,5)
                            m = (10-5)/(4000-900)
                            area = stiker_size[0]*stiker_size[1]
                            text_size = m*(area-4000) + 10

                            # Don't make smaller than size 6
                            text_size = max(text_size, 6)

                            # Show the name of the color of each sticker
                            ax.annotate(color_names[face_list[i]], face_positions[:,i], color='white', size=text_size, ha='center')



                elif not self.solution_found:
                    ## Phase 2: Solve cube in a thread

                    if self.solver is None:
                        # Launch the solver thread

                        # Arrange faces aquired into a cube
                        self.cube = self.arrange_cube(faces, colors_checked)

                        # Launch solver thread
                        self.solver = Cube_solver_thread(self.cube, self.solver_type)
                        self.solver.setDaemon(True)
                        self.solver.start()

                        #
                        solution_timer = time.time()

                    if self.solver.solution_found:
                        # The solver has found a solution, go to the next phase
                        plt.title("Showing cube solution")

                        self.solution_found = True

                        # Show solution on the console
                        print("Solution in text form:")
                        print(self.solver.solution)
                        print(f"It took {time.time() - solution_timer} seconds to find the solution")

                        # Set up the solution list for the next phase
                        solution = self.solver.solution
                        solution.append("--")

                        # Store a cube with one of the moves done for next phase
                        self.cube_next = self.cube.turn(solution[0])
                    else:
                        # The solver is not done yet, be nice to the user, they might get impatient
                        plt.title("Solving cube...")

                        # Show rotating section of a circle
                        indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, 0.2, (time.time() - solution_timer)*180)
                        indicator_color = "white"
                elif solution[0] != 'Incorrect cube solver' and solution[0] != '--':
                    ## Phase 3: Show solution steps

                    if self.debug:
                        print(self.cube.toStringColor())
                        print(self.cube_next.toStringColor())

                    plt.title("Showing solution")

                    # Check for colors that would never be on the stickers of a rubik's cube
                    has_bad_colors = False
                    if face.ndim == 2:
                        has_bad_colors = (face < 0).any()
                        if has_bad_colors:
                            print("Bad colors found")

                    if face.ndim == 2 and not has_bad_colors:
                        # We are looking at a face right now

                        # The color of the center tells us which face we are looking at
                        face_color = int(face[1,1])

                        # We get the rotation we would have to do to a normalized cube for
                        # it to be oriented like our cube
                        transform, unique = self.cube.face_to_front(face)

                        # We apply the transformation to the first move of the solution
                        relative_move = transformAlg([solution[0]], transform)[0]


                        if relative_move[0] not in ['B', 'F'] and face_equal_norot(self.cube_next.faces[face_color,:,:], face):
                            # Show progress circle to confirm that the face has been turned correctly
                            indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, 2*(time.time() - face_confirm_timer)/self.confirm_time, 0)
                            indicator_color = "orange"

                            # If the timer is done, go to the next step of the solution
                            if time.time() - face_confirm_timer > self.confirm_time/2:
                                self.cube = self.cube.turn(solution.pop(0))
                                self.cube_next = self.cube_next.turn(solution[0])

                        elif face_equal_norot(self.cube.faces[face_color,:,:], face):
                            # Reset the confirmation timer
                            face_confirm_timer = time.time()

                            # We initiate some parameters
                            face_pos_aux = face_positions[[1,0],:]
                            rot = 0
                            indic_size = 80
                            if relative_move[-1] == "'":
                                rot = 180


                            # We show an arrow indicating the next move that the user must perform
                            arrow_pos = [0,0]
                            if relative_move[0] == 'U':
                                arrow_pos = face_pos_aux[:,1]
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, arrow_pos, 0, (180+rot)%360)
                                arrow_pos = arrow_pos + np.array([-40,0])
                            elif relative_move[0] == 'D':
                                arrow_pos = face_pos_aux[:,7]
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, arrow_pos, 0, (rot)%360)
                                arrow_pos = arrow_pos + np.array([-40,0])
                            elif relative_move[0] == 'R':
                                arrow_pos = face_pos_aux[:,5]
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, arrow_pos, 0, 90 + rot)
                                arrow_pos = arrow_pos + np.array([0,40])
                            elif relative_move[0] == 'L':
                                arrow_pos = face_pos_aux[:,3]
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, arrow_pos, 0, 270 - rot)
                                arrow_pos = arrow_pos + np.array([0,40])
                            elif relative_move[0] == 'F':
                                arrow_pos = face_pos_aux[:,4]
                                indicator_x, indicator_y = self.get_shape_coords("arrow_circle", indic_size, arrow_pos, 0, rot)
                                arrow_pos = arrow_pos + np.array([120,0])
                            elif relative_move[0] == 'B':
                                arrow_pos = face_pos_aux[:,4]
                                indicator_x, indicator_y = self.get_shape_coords("arrow_circle", indic_size, arrow_pos, 0, (180+rot)%360)
                                arrow_pos = arrow_pos + np.array([120,0])
                                linestyle = "dashed"
                            indicator_color = "#66ccff"

                            if relative_move[-1] == "2":
                                ax.annotate("x2", (arrow_pos[1], arrow_pos[0]), color='white', size=10, ha='center')


                            if relative_move[0] in ['B', 'F']:
                                # We cannot work with B or F moves, performing them doesn't change the face making it
                                # imposible to guide the user towards the solution
                                ax.annotate("Change face please", face_positions[:,4], color='white', size=10, ha='center')


                    else:
                        # Reset confirmation timer
                        face_confirm_timer = time.time()

                    # Show the entire solution in the screen
                    ax.annotate(" ".join(solution[:-1]), (indicator_offset[1], indicator_offset[0]),  color='white', size=15, ha='center')

                else:
                    ## END: We are done, close the app
                    print("CONGRATULATIONS, YOUR CUBE IS NOW SOLVED.")
                    self.finished = True

                # Draw indicator
                ax.plot(indicator_x, indicator_y, indicator_color, linewidth = 6, linestyle = linestyle)


                ## Display
                im = ax.imshow(np.zeros(I_rgb.shape))
                im.set_data(frame_original)

                # Update plot
                fig.canvas.draw_idle()
                fig.canvas.flush_events()


                # Measure time for FPS control
                frame_end = time.time()

                # Limit FPS
                time_passed = frame_end - frame_start
                if time_passed < frame_time:
                    time.sleep(frame_time-time_passed)


        except KeyboardInterrupt:
            print("Manually interrupted")
        except Exception:
            traceback.print_exc()

        if solver_thread is not None:
            solver_thread.join()

        self.cam.stop()
        self.current_face = -1
