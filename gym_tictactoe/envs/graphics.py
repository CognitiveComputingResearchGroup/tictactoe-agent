import pygame

mark_img = {'X': pygame.image.load("tic.jpg"),
            'O': pygame.image.load("tac.jpg")
           }
bkg_x_offset = 50
bkg_y_offset = 50
board_x_offset = 10
board_y_offset = 10

def reset():
    img = pygame.image.load("board.jpg")
    screen = pygame.display.set_mode((1000,1000))
    screen.blit(img, (bkg_x_offset,bkg_y_offset))
    pygame.display.flip()
    return screen


def initialize():
    pygame.init()

    screen = reset()

    # define a variable to control the main loop
    running = True

    '''
    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False
    '''

    pygame.display.flip()
    return screen

screen = initialize()

def draw(board):
    reset()
    for i, b in enumerate(board):
        if b != 0:
            put({1: 'X', -1: 'O'}[b], i+1)

def put(mark, pos, screen=screen):
    pos = pos-1
    x = (pos%3)*200 + bkg_x_offset + board_x_offset
    y = int(pos/3)*200 + bkg_y_offset + board_y_offset
    screen.blit(mark_img[mark], (x,y))
    pygame.display.flip()

def main():

    screen = initialize()

    put('X',1,screen)
    put('X',2,screen)
    put('X',3,screen)
    put('O',4,screen)
    put('O',5,screen)
    put('O',6,screen)
    put('O',7,screen)
    put('X',8,screen)
    put('X',9,screen)


    # define a variable to control the main loop
    running = True

    # main loop
    while running:
        # event handling, gets all event from the event queue
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                running = False


if __name__ == "__main__":
    main()