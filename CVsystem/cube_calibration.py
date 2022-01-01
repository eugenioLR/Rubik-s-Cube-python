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
    are_equal = False
    i = 0
    while i < 4 and not are_equal:
        are_equal = are_equal or (face1 == np.rot90(face2, i)).all()
        i += 1

    return are_equal


class Cube_calibrator:
    def __init__(self, fps=30, confirm_time=2.5):
        self.last_face = -1

        self.calibrated = False
        self.solution_found = False
        self.finished = False
        self.solver_type = 'IDA*-NN'

        self.cube = Cube(3)
        self.cube_next = None
        self.cam = WebcamVideoStream(fps=fps)

        self.fps = fps
        self.confirm_time = confirm_time
        self.tick_time = 0.5

        self.solver = None

    def get_shape_coords(self, shape_type, scale, offset, progress, rotation):
        if shape_type == 'circle':
            rad_rot = np.pi*rotation/180
            max_theta = (2*np.pi)*progress
            theta = np.linspace(0, max_theta, 100)
            theta = theta + rad_rot
            indicator_x = scale*np.cos(theta) + offset[1]
            indicator_y = scale*np.sin(theta) + offset[0]
        elif shape_type == 'tick':
            indicator_x = scale*np.array([-0.5,0,0.75]) + offset[1]
            indicator_y = scale*np.array([0,0.75,-0.75]) + offset[0]
        elif shape_type == 'arrow':
            indicator_x = scale*np.array([-1,1,0.25,1,0.25])
            indicator_y = scale*np.array([0, 0, 0.5,0,-0.5])

            if rotation == 90:
                indicator_x, indicator_y = indicator_y, -indicator_x
            elif rotation == 180:
                indicator_x = -indicator_x
            elif rotation == 270:
                indicator_x, indicator_y = indicator_y, indicator_x

            indicator_x = indicator_x + offset[1]
            indicator_y = indicator_y + offset[0]
        elif shape_type == 'arrow_circle':
            #rad_rot = np.pi*rotation/180
            theta = np.linspace(0, 3*np.pi/2, 100)
            #theta = theta + rad_rot
            indicator_x = scale*np.hstack([np.cos(theta), [0.2, -0.2, 0.2, -0.2, 0.2]])
            indicator_y = scale*np.hstack([np.sin(theta), [-1, -0.5, -1, -1.5, -1]])
            if rotation == 180:
                indicator_x = -indicator_x

            indicator_x = indicator_x + offset[1]
            indicator_y = indicator_y + offset[0]
        elif shape_type == 'dash':
            indicator_x = scale*np.array([-1,1]) + offset[1]
            indicator_y = scale*np.array([0, 0]) + offset[0]

        return indicator_x, indicator_y

    def calibrate_cube(self, phone=True):
        self.cam.start()
        solver_thread = None
        try:
            # Initialize plot
            plt.ion()
            fig = plt.figure()
            fig.canvas.mpl_connect('close_event', on_close)
            ax = plt.subplot()
            path = str(Path(__file__).resolve().parent) + "/"
            I_rgb = cv2.imread(path+"rubiks_cube_photo.png")
            im = ax.imshow(np.zeros(I_rgb.shape))
            indic, = ax.plot([0,0],[0,0], linewidth=10)
            plt.show()

            # Set indicator offset
            indicator_offset = [I_rgb.shape[0]*(7/8), I_rgb.shape[1]/2]

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

                if phone:
                    frame_original = cv2.rotate(frame_original, cv2.ROTATE_90_CLOCKWISE)


                ## Pre-processing
                frame_hsv = cv2.cvtColor(frame_original, cv2.COLOR_RGB2HSV)

                #frame = cv2.fastNlMeansDenoisingColored(frame, None, 10,10,7,21) # Too slow for real time, but gives the best results
                frame_hsv = cv2.medianBlur(frame_hsv, 9)
                #frame_hsv[:,:,2] = cv2.equalizeHist(frame_hsv[:,:,2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)


                ## Extraction of characteristics
                frame_bw = binarize(frame_hsv)
                contours, positions = find_contours(frame_bw, debug = False)


                ## Description
                face, face_positions = get_ordered_colors(frame_hsv, contours, debug = False)

                self.calibrated = True
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
                        #indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, 0.2, 180*(time.time() - face_confirm_timer)/self.confirm_time)
                        indicator_color = "orange"

                        # When the timer finishes add the face to out list of faces
                        if time.time() - face_confirm_timer > self.confirm_time:
                            colors_checked.append(face[1,1])
                            faces.append(face)
                            #print("new face detected")
                            #print(face)
                            self.last_face = face[1,1]
                            if len(faces) == 6:
                                self.calibrated = True

                    elif face.ndim == 2 and (face == prev_face).all() and face[1,1] in colors_checked and time.time() - tick_timer < self.tick_time:
                        # The face we are seeing is stored

                        # Reset timer for face confirmation
                        face_confirm_timer = time.time()

                        # Get green check mark to draw later
                        indicator_x, indicator_y = self.get_shape_coords("tick", indic_size, indicator_offset, 0, 0)
                        indicator_color = "green"
                    else:
                        # Direct user to the next face to check

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
                            ax.annotate(color_names[face_list[i]], face_positions[:,i], color='white', size=10, ha='center')



                elif not self.solution_found:
                    ## Phase 2: Solve cube in a thread

                    if self.solver is None:
                        # Launch the solver thread
                        #self.cube = Cube(3).doAlgorithm(['U2', 'B2', 'R', 'L'])
                        self.cube = Cube(3).doAlgorithm(['U\'', 'F\''])
                        #self.cube = self.arrange_cube(faces, colors_checked)
                        self.solver = Cube_solver_thread(self.cube, self.solver_type)
                        self.solver.setDaemon(True)
                        self.solver.start()
                        solution_timer = time.time()

                    if self.solver.solution_found:
                        # The solver has found a solution, go to the next phase
                        plt.title("Showing cube solution")
                        print("nice")
                        print(self.solver.solution)
                        solution = self.solver.solution
                        solution.append("--")
                        self.solution_found = True
                        self.cube_next = self.cube.turn(solution[0])
                    else:
                        # The solver is not done yet, be nice to the user, they might get impatient
                        plt.title("Solving cube...")

                        # Show rotating section of a circle
                        indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, 0.2, (time.time() - solution_timer)*180)
                        indicator_color = "white"
                elif solution[0] != 'Incorrect cube solver' and solution[0] != '--':
                    ## Phase 3: Show solution steps

                    plt.title("Showing solution")

                    # TODO: the current face shows the orientation, this face is the "front" face
                    # we know the color is face[1,1] adn we can test the 4 rotations to know how out cube is oriented
                    # with something like an "inverse normalization"

                    has_bad_colors = False

                    if face.ndim == 2:
                        has_bad_colors = (face < 0).any()
                        if has_bad_colors:
                            print("Bad colors found")

                    if face.ndim == 2 and not has_bad_colors:

                        face_color = int(face[1,1])

                        transform, unique = self.cube.face_to_front(face)


                        relative_move = transformAlg([solution[0]], transform)[0]
                        #print(f"face of color {color_names[int(face[1,1])]} found")



                        if relative_move[0] not in ['B', 'F'] and face_equal_norot(self.cube_next.faces[face_color,:,:], face):
                            # The user has performed the correct move

                            indicator_x, indicator_y = self.get_shape_coords("circle", indic_size, indicator_offset, 2*(time.time() - face_confirm_timer)/self.confirm_time, 0)
                            indicator_color = "orange"

                            if time.time() - face_confirm_timer > self.confirm_time/2:
                                self.cube = self.cube.turn(solution.pop(0))
                                self.cube_next = self.cube_next.turn(solution[0])

                        elif face_equal_norot(self.cube.faces[face_color,:,:], face):
                            face_confirm_timer = time.time()

                            face_pos_aux = face_positions[[1,0],:]
                            rot = 0
                            indic_size = 80
                            if relative_move[-1] == "'":
                                rot = 180



                            # We have to show the user the correct turn to make
                            if relative_move[0] == 'U':
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, face_pos_aux[:,1], 0, (180+rot)%360)
                            elif relative_move[0] == 'D':
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, face_pos_aux[:,7], 0, (rot)%360)
                            elif relative_move[0] == 'R':
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, face_pos_aux[:,5], 0, 90 + rot)
                            elif relative_move[0] == 'L':
                                indicator_x, indicator_y = self.get_shape_coords("arrow", indic_size, face_pos_aux[:,3], 0, 270 - rot)
                            elif relative_move[0] == 'F':
                                indicator_x, indicator_y = self.get_shape_coords("arrow_circle", indic_size, face_pos_aux[:,4], 0, rot)
                            elif relative_move[0] == 'B':
                                indicator_x, indicator_y = self.get_shape_coords("arrow_circle", indic_size, face_pos_aux[:,4], 0, (180+rot)%360)
                                linestyle = "dashed"
                            indicator_color = "#66ccff"


                            if relative_move[0] in ['B', 'F']:
                                ax.annotate("Change face please", face_positions[:,4], color='white', size=10, ha='center')


                    else:
                        face_confirm_timer = time.time()


                    ax.annotate(" ".join(solution[:-1]), (indicator_offset[1], indicator_offset[0]),  color='white', size=15, ha='center')


                else:
                    # Phase 4: We are done, close the app
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
            pass
        except Exception:
            traceback.print_exc()
        if solver_thread is not None:
            solver_thread.join()
        self.cam.stop()
        self.current_face = -1

    def arrange_cube(self, faces, colors):
        opposites = {0:5,1:3,2:4,3:1,5:0}

        faces[0] = np.rot90(faces[0],3)
        faces[5] = np.rot90(faces[5],2)

        return Cube(3, faces)






if __name__ == '__main__':
    cubeCal = Cube_calibrator(60, 2.5)
    cubeCal.calibrate_cube()
