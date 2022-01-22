#@title Generate video using ffmpeg

init_frame =  1#@param  {type: "number"}
last_frame =  25#@param {type: "number"}

min_fps =  60#@param {type: "number"}
max_fps =  60#@param {type: "number"}

total_frames = last_frame-init_frame

# Desired video runtime in seconds
length =  1#@param {type: "number"}
use_upscaled_images = False #@param {type: "boolean"}
frames = []
tqdm.write('Generating video...')

if use_upscaled_images == True:
  for filename in os.listdir(ESRGAN_path + "/results/"):
      filename = f"{ESRGAN_path}/results/{filename}"
      frames.append(Image.open(filename))
elif use_upscaled_images == False:
  for i in range(init_frame,last_frame): #
    if usingDiffusion == False:
      filename = f"{abs_root_path}/vqgan-steps/{i:04}.png"
      frames.append(Image.open(filename))
    elif usingDiffusion == True:
      filename = f"{abs_root_path}/diffusion-steps/{i:05}.png"
      frames.append(Image.open(filename))

#fps = last_frame/10
fps = np.clip(total_frames/length,min_fps,max_fps)

# Names the video after the prompt if there is one, if not, defaults to video.mp4
def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 += ele  
    
    # return string  
    return str1 
        

video_filename = "video" #@param {type: "string"}
#@markdown Note: using images previously upscaled by ESRGAN may take longer to generate


video_filename = listToString(video_filename).replace(" ","_")
print("Video filename: "+ video_filename)

video_filename = video_filename + ".mp4"

from subprocess import Popen, PIPE
p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', video_filename], stdin=PIPE)
for im in tqdm(frames):
    im.save(p.stdin, 'PNG')
p.stdin.close()

print("Compressing video...")
p.wait()


print("Video ready.")