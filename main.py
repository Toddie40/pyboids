import pygame
import numpy as np


class Color:
    colors = {
        'white': (255,255,255),
        'black': (40,40,40),
        'blue': (119, 158, 203),
        'red': (255, 105, 96),
        'green': (119, 221, 119),
        'purple': (199, 206, 238),
        'yellow': (253, 253, 150)
    }

    foreground = [colors['blue'], colors['white'], colors['red'], colors['green'], colors['purple'], colors['yellow']]

    def getRandomForegroundColor():
        i = np.random.randint(0, len(Color.foreground))
        return Color.foreground[i]

class Display:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.initialise()

    def initialise(self):
        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.Clear()
    
    def Clear(self):
        self.screen.fill(Color.colors['black'])

    def drawBoid(self, position, rotation, scale, color):
        
        position = position.transpose()
        rotation = rotation.transpose()

        rotation90 = np.array(  [[0, -1],
                                [1, 0]])

        p1 = position + scale * rotation
        p2 = position + scale / 3 * np.matmul(rotation90, rotation)
        p3 = position - scale / 3 * np.matmul(rotation90, rotation)

        p1 = p1.transpose()
        p2 = p2.transpose()
        p3 = p3.transpose()

        pygame.draw.polygon(self.screen, color, (tuple(p1[0]), tuple(p2[0]), tuple(p3[0])))

class BoidSpace:

    def __init__(self, x, y):
        self.size = np.array([x,y]) #tuple describing the size of the boid space e.g. (10,10) wsould be a 10x10 grid
        self.boids = [] # lift of boids in the boid space
        self.boidVision = 150.0
        self.boidSpeed = 0.4
        self.rotationSpeed = 50
        self.delta_time = 0.1
        self.fps = 60

        self.alignmentFactor = 1
        self.cohesionFactor = 1
        self.separationFactor = 1


        self.totalFactors = self.alignmentFactor + self.cohesionFactor + self.separationFactor
        
        self.alignmentWeighting = self.alignmentFactor / self.totalFactors
        self.cohesionWeighting = self.cohesionFactor / self.totalFactors
        self.separationWeighting = self.separationFactor / self.totalFactors

    def AddBoid(self, boid):
        self.boids.append(boid)

    def AddRandomBoid(self):
        position = np.random.random_sample([1,2]) * self.size
        rotation = np.random.random_sample([1,2])
        rotation = rotation / np.linalg.norm(rotation) #normalise the rotation vector to be of size 1
        
        #add random variance of 80-120% of speed value
        randomSpeedModifier = (.8 + np.random.rand()/2.5)
        randomRotationModifier = (.8 + np.random.rand()/2.5)
        randomVisionModifier = (.8 + np.random.rand()/2.5)

        b = Boid(self, position, rotation, self.boidSpeed * randomSpeedModifier, self.boidVision * randomVisionModifier, self.rotationSpeed * randomRotationModifier)
        self.boids.append(b)
        return b

    def Update(self):
        # calculate time since last frame
        self.delta_time = pygame.time.Clock().tick(self.fps)        
        # tell boids to update

        for boid in self.boids:
            boid.Update(self.delta_time)

    def GetBoids(self):
        b = self.boids
        return b

class Boid:
    # a boid will always move forwards 
    # i.e facing its rotation vector at a speed of speed.
    # it will be able to collect information about its 
    # surroundings up to the range of its vision
    def __init__(self, space, position: list, rotation: list, speed: float, visionRange: float, rotationSpeed: float):
        if np.shape(position) == (1,2) and np.shape(rotation) == (1,2):
            self.position = np.array(position)
            self.rotation = np.array(rotation)
        else:
            print("boids must be created with 2d rotation and position vectors")
            return None
        self.speed = speed
        self.rotationSpeed = rotationSpeed
        self.vision = visionRange
        self.space = space
        self.color = Color.getRandomForegroundColor()
    
    def Update(self, delta_time):
        
        closeBoids = self.GetCloseBoids()
        noCloseBoids = len(closeBoids) - 1
        angleToRotate = 0.0
        if noCloseBoids > 0:
            # calculate average position and direction
            totalPositions = np.array([[0.0,0.0]])
            totalRotation = np.array([[0.0, 0.0]])
            totalWeightedSeparation = np.array([[0.0, 0.0]])
            
            for b in closeBoids:
                if (b != self):
                    totalPositions += b.position
                    totalRotation += b.rotation
                    separation = (b.position - self.position)   
                    weightedSeparation = separation
                    dist = np.linalg.norm(separation)
                    totalWeightedSeparation += weightedSeparation / (dist*dist) 

            
            averagePosition = totalPositions / noCloseBoids
            averageRotation = totalRotation / noCloseBoids
            averageWeightedSeparation = -totalWeightedSeparation / noCloseBoids # invert direciton so it points away from average boid position weighted towards close boids
            averageWeightedSeparation /= np.linalg.norm(averageWeightedSeparation)

            # alignment
            # cohesion
        
            directionToAveragePosition = averagePosition - self.position

            length = np.linalg.norm(directionToAveragePosition)

            if length != 0: # if the average position is the same as this boids position then there is nothing in range except itself. We can't divide by zero so we do this check
                directionToAveragePosition /= length

                rotation3D = np.append(self.rotation, 0).transpose()
                directionToAveragePosition = np.append(directionToAveragePosition, 0).transpose()
                averageRotation = np.append(averageRotation, 0).transpose()
                separationDirection = np.append(averageWeightedSeparation, 0).transpose()

                # calculate cross products between direction vectors to get the angle to rotate towards.
                angleToAveragePosition = np.arcsin(np.cross(rotation3D, directionToAveragePosition))
                angleToAverageRotation = np.arcsin(np.cross(rotation3D, averageRotation))     
                angleToSeparation = np.arcsin(np.cross(rotation3D, separationDirection))     
                
                idealRotationAngle = (angleToAveragePosition * self.space.cohesionWeighting + angleToAverageRotation*self.space.alignmentWeighting + angleToSeparation*self.space.separationWeighting) / 3

                angleToRotate = float(idealRotationAngle[2] * delta_time/1000 * self.rotationSpeed)
        
                
            
        # create a rotation matrix to transform towards this angle
        costheta = np.cos(angleToRotate)
        sintheta = np.sin(angleToRotate)
        rotationMatrix = np.array([ [costheta, -sintheta],
                                    [sintheta, costheta]])
        

        # a boid does 3 things
        # separation
        
        
        #apply transformation matrices

        self.rotation = np.matmul(rotationMatrix, self.rotation.transpose()).transpose()
       
        #move boid
        self.position = self.position + self.rotation * self.speed * delta_time
        
        #handle edge of screen stuff
        if (self.position[0][0] > self.space.size[0] or self.position[0][0] < 0):
            self.rotation[0][0] *= -1
        if (self.position[0][1] > self.space.size[1] or self.position[0][1] < 0):
            self.rotation[0][1] *= -1


        return True
    def GetCloseBoids(self):
        # get close boids
        closeBoids = []
        for b in self.space.boids:
            disp = b.position - self.position
            separation = np.linalg.norm(disp)
            if separation < self.vision:
                closeBoids.append(b)
        return closeBoids

if __name__ == "__main__":

    width = 1280
    height = 720
    boidSize = 20
    disp = Display(width,height)
    space = BoidSpace(width,height)
    for x in range(0,50):
        space.AddRandomBoid()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        #update boids
        space.Update()
        
        #clear display
        disp.Clear()
        #draw them
        for boid in space.boids:
            disp.drawBoid(boid.position, boid.rotation, boidSize, boid.color)


        pygame.display.update()
    #do stuff