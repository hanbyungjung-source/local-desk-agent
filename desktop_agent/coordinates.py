from game_agent.core import Region


class PixelRegion(Region):
    def __init__(self,region,frame):
        super().__init__(region.left,region.top,region.width,region.height)
        self.frame = frame

    def point(self,x,y):
        width,height = self.frame['image_size']
        if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
            raise ValueError('Pixel coordinates are outside the reference image')
        left,top,right,bottom = self.frame['bounds']
        horizontal = left+round(x*(right-left-1)/max(1,width-1))
        vertical = top+round(y*(bottom-top-1)/max(1,height-1))
        if not (self.left <= horizontal < self.left+self.width and self.top <= vertical < self.top+self.height):
            raise ValueError('Image point is outside the selected window')
        return horizontal,vertical


def input_region(region,arguments,frame,window):
    if arguments.get('coordinate_space')!='image_pixels':
        return region
    if frame is None:
        frame = dict(bounds=region.bbox,image_size=[region.width,region.height],source='selected_window',
                     handle=window['handle'],pid=window['pid'])
    if frame.get('source') not in ('selected_window','visible_desktop'):
        raise ValueError('This image is not a desktop coordinate reference; capture the target window')
    if frame.get('handle') != window['handle'] or frame.get('pid') != window['pid']:
        raise ValueError('Coordinate reference belongs to a different selected window')
    expected = frame.get('target_bounds',frame['bounds'])
    if tuple(expected)!=tuple(region.bbox):
        raise ValueError('Selected window moved or resized; refresh the coordinate reference')
    mapped = PixelRegion(region,frame)
    for horizontal,vertical in (('x','y'),('end_x','end_y')):
        if horizontal in arguments and vertical in arguments:
            mapped.point(arguments[horizontal],arguments[vertical])
    return mapped