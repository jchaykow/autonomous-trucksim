from imports import *
from key_presses import *

class record_frames():
    def __init__(self, file_name=None):
        if file_name is None: file_name = 'training_data_{date}.npy'.format(date = datetime.now().strftime("%m%d%Y"))

        self.training_data = [] # initialize training_data
        self.keys = ["'w'"] # initialize keys
        
        self.file_name = file_name
    
    def record(self):
        for i in list(range(4))[::-1]:
            print(i+1)
            time.sleep(1)
        
        with mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 79, "left": 265, "width": 905, "height": 586}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                screen = np.array(sct.grab(monitor))

                print("fps: {}".format(1 / (time.time() - last_time)))

                # screen =  np.array(ImageGrab.grab(bbox=(265 * 2,79 * 2,1170 * 2,665 * 2)))
                last_time = time.time()

                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                screen = cv2.resize(screen, (224,224))
                # cv2.imshow('window2',cv2.cvtColor(cv2.resize(original_image, (800,600)), cv2.COLOR_BGR2RGB))

                key_check()
                print([self.keys[-1]])
                output = keys_to_output([self.keys[-1]])
                self.training_data.append([screen,output])

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                if len(self.training_data) % 10 == 0:
                    print(len(self.training_data))
                    np.save(self.file_name,self.training_data)
    
    def balance_data(self):
        lefts = []
        rights = []
        forwards = []

        shuffle(self.training_data)

        for data in self.training_data:
            img = data[0]
            choice = data[1]
            
            if choice == [1,0,0]:
                lefts.append([img,choice])
            elif choice == [0,1,0]:
                forwards.append([img,choice])
            elif choice == [0,0,1]:
                rights.append([img,choice])
            else:
                print('no matches')

        forwards = forwards[:len(lefts)][:len(rights)]
        lefts = lefts[:len(forwards)]
        rights = rights[:len(forwards)]

        final_data = forwards + lefts + rights
        shuffle(final_data)

        return final_data
