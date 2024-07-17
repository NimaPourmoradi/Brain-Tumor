<img src="https://i.postimg.cc/ydZcb9cY/Hello.jpg">

<img src="https://i.postimg.cc/QdHhq8qX/Brain-Tumor-2-25-2024.png">

<img src="https://i.postimg.cc/nzqPFBVd/Brain-MRI.jpg">

<div style="background-color:#e1dee3; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);border:2px solid #0A2342">
    <h1 style="font-size:24px; font-family:calibri; color:#0A2342;"><b>Notebook Structure</b></h1>
    <p style="font-size:20px; font-family:calibri; line-height: 1.5em; text-indent: 20px;">In this project, mainly I use keras and cv2 to main structure and matplotlib and seaborn to visualization. This is Deep Learning project that use Convolutional Neural Network to work on images and make a accurate model.</p>
    <p style="font-size:20px; font-family:calibri; line-height: 1.5em; text-indent: 20px;"> We start with reading and loadin dataset. Then apply some preprocessing on dataset and images and then create a model to train. when we reach acceptable evaluation of model, make a prediction part with ipywidget (Button ux).
</div>

<div style="background-color:#e1dee3; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);border:2px solid #0A2342">
    <h1 style="font-size:24px; font-family:calibri; color:#0A2342;"><b>Context</b></h1>
    <p style="font-size:20px; font-family:calibri; line-height: 1.5em; text-indent: 20px;">In this project, we have a image dataset of about 3096 Brain MRI image that classified in 4 classes ('normal', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor').
    Our target is to make a classification model to detect kind of brain tumor in MRIs.
    </p>
</div>

# <div style="background-color:#0A2342;background-size: cover;font-family:tahoma;font-size:175%;text-align:center;border-radius:25px 50px; padding:10px; border:solid 2px #09375b"><span style="color:#f2f2f2"><b>Table Of Content</b></span></div>

<a id='tbl_content'></a>
<div style="background-color:#eef1fb; padding: 20px 10px 10px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1)">
    <ul>
        <li><a href="#step1" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 1 | Initila Configurations </a></li>
            <ul>
                <li><a href="#step11" style="font-size:18px; font-family:calibri"> Step 1.1 | Import Python Libraries </a></li>
                <li><a href="#step12" style="font-size:18px; font-family:calibri"> Step 1.2 | Colors </a></li>
            </ul>
        <li><a href="#step1" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 2 | Load Data </a></li>
            <ul>
                <li><a href="#step21" style="font-size:18px; font-family:calibri" > Step 2.1 | Load and Read Data </a></li>
                <li><a href="#step22"> Step 2.2 | Plot and Count Images </a></li>
            </ul>
        <li><a href="#step3" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 3 | Data Preprocessing </a></li>
            <ul>
                <li><a href="#step31" style="font-size:18px; font-family:calibri"> Step 3.1 | Create Tesorflow Dataset </li>
                <li><a href="#step31" style="font-size:18px; font-family:calibri"> Step 3.2 | Create Train, Validation and Test Dataset </li>
            </ul>
        <li><a href="#step4" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 4 | Modeling </a></li>
            <ul>
                <li><a href="#step41" style="font-size:18px; font-family:calibri"> Step 4.1 | Pre-Trained Model </a></li>
                <li><a href="#step42" style="font-size:18px; font-family:calibri"> Step 4.2 | Freeze Pre-Trained Model Layers </a></li>
                <li><a href="#step43" style="font-size:18px; font-family:calibri"> Step 4.3 | Define Layers For Join Pre-Trained Model </a></li>
                <li><a href="#step44" style="font-size:18px; font-family:calibri"> Step 4.4 | Compile Model </a></li>
                <li><a href="#step45" style="font-size:18px; font-family:calibri"> Step 4.5 | Call-Backs </a></li>
                <li><a href="#step46" style="font-size:18px; font-family:calibri"> Step 4.6 | Train Model </a></li>
                <li><a href="#step47" style="font-size:18px; font-family:calibri"> Step 4.7 | Evaluation </a></li>
            </ul>
        <li><a href="#step5" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 5 | Predict </a></li>
        <li><a href="#author" style="font-size:24px; font-family:calibri; font-weight:bold"> Step 6 | Author </a></li>
    </ul>

</div>

<div style="background-color:#c5d8d1; padding: 0px 0px 0px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:left">
    <p style="font-size:70px; font-family:tahoma; line-height: 2em; text-indent: 10px; font-weight:bolder; color:#0A2342">Lets Go !
    </p>
</div>

# <a id='step1'></a>
# <div style="background-color:#0A2342;background-size: cover;font-family:tahoma;font-size:175%;text-align:center;border-radius:25px 50px; padding:10px; border:solid 2px #09375b"><span style="color:red"><b>Step 1 | <b></span><span style="color:#f2f2f2"><b>Initila Configurations</b></span></div>

##### [🏠 Tabel of Contents](#tbl_content)

<a id='step11'></a>
# <span style="font-family:tahoma;font-size:100%;text-align:left"><span style="color:red"><b>Step 1.1 | <b></span><span style="color:#1d84b5"><b>Import Python Libraries</b></span></span>

<div style="background-color:#f4edea; padding: 20px 10px 10px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:left">
    <p style="font-size:18px; font-family:tahoma; line-height: 2em; text-indent: 20px;">➡️ At first, to avoid tensorflow and python warnings, use <b>silence_tensorflow</b> and <b>warnings</b> library. Because silence_tensorflow not installed in kaggle by default, try to install it with <b>pip</b> comand.
    </p>
    <p style="font-size:18px; font-family:tahoma; line-height: 1.5em; text-indent: 20px;">➡️ Then import necessary python-libraries that install before with <code>pip install command</code> comand.
    </p>
</div>

<a id='step47'></a>
# <span style="font-family:tahoma;font-size:100%;text-align:left"><span style="color:red"><b>Step 4.7 | <b></span><span style="color:#1d84b5"><b>Evaluation</b></span></span>


```python
# checkpoint callback, save base model weights in "MyModel.keras".
# So, we should load it by keras.models.load_model
best_model = keras.models.load_model('MyModel.keras')
```


```python
# Evaluate model by model.evaluate()
loss, accuracy = best_model.evaluate(test_ds)
print()
print(colored(f'Loss : {loss}', 'green', attrs=['bold']))
print(colored(f'Accuracy : {accuracy*100}%', 'green', attrs=['bold']))
```

    [1m10/10[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 98ms/step - accuracy: 1.0000 - loss: 9.5755e-05
    
    [1m[32mLoss : 7.377217843895778e-05[0m
    [1m[32mAccuracy : 100.0%[0m
    

<a id='step5'></a>
# <div style="background-color:#0A2342;background-size: cover;font-family:tahoma;font-size:175%;text-align:center;border-radius:25px 50px; padding:10px; border:solid 2px #09375b"><span style="color:red"><b>Step 5 | <b></span><span style="color:#f2f2f2"><b>Predict</b></span></div>

##### [🏠 Tabel of Contents](#tbl_content)

<div style="background-color:#f4edea; padding: 25px 10px 10px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:left">
    <p style="font-size:18px; font-family:tahoma; line-height: 2em; text-indent: 20px;">➡️ I've made these Widgets in which we can upload images from our local machine and predict whether the MRI scan has a Brain Tumour or not and to classify which Tumor it is. Unfortunately, it doesn't work on Kaggle but you can play around with this by downloading the notebook on your machine :)
    </p>
</div>


```python
# A function to load uploaded image, pre-processing and predict model
def predict_image(upload) :
    # Store a location in memory that image stored(with uploader)
    uploaded_content = list(upload.value)[0].content
    # Load image from memory to img variable by io.BytesIO and PIL.Image
    img = Image.open(io.BytesIO(uploaded_content))
    # Load image by cv2
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Resize image to (224, 224) whitch is ResNet50V2 input size
    img = cv2.resize(opencvImage,(224,224))
    # Reshape image array to add 3 channels
    img = img.reshape(1,224,224,3)
    # Predict model with best_model
    p = best_model.predict(img)
    # Return high probability of predictions
    p = np.argmax(p,axis=1)[0]
    
    
    if p==0:
        print('The model predicts that there is no tumor')
        
    elif p==1:
        p='Pituitary Tumor'
        
    elif p==2:
        p='Meningioma Tumor'
        
    if p==3:
        p='Glioma Tumor'
    
    if p!=0:
        print()
        print(colored(f'The Model predicts that it is a {p}', 'green'))
```

<div style="background-color:#f4edea; padding: 25px 10px 10px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:left">
    <p style="font-size:18px; font-family:tahoma; line-height: 2em; text-indent: 20px;">➡️ You can upload your MRI image from your Hard-disk by clicking on button below. 
    </p>
</div>

After uploading the image, you can click on the Predict button below to make predictions:


```python
# Create Upload button by ipywidget

upload = FileUpload(accept='.jpg', multiple=False)
display(upload)
```


    FileUpload(value=(), accept='.jpg', description='Upload')


<div style="background-color:#f4edea; padding: 25px 10px 10px 10px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:left">
    <p style="font-size:18px; font-family:tahoma; line-height: 2em; text-indent: 20px;">➡️ After uploading the image, you can click on the Predict button below to make predictions:
</div>


```python
button = widgets.Button(description='Predict')
out = widgets.Output()
def on_button_clicked(_):
    with out:
        clear_output()
        try:
            predict_image(upload)
            
        except:
            print('No Image Uploaded/Invalid Image File')
button.on_click(on_button_clicked)
widgets.VBox([button,out])
```




    VBox(children=(Button(description='Predict', style=ButtonStyle()), Output()))



<a id="author"></a>
<div style="border:3px solid navy; border-radius:30px; padding: 15px; background-size: cover; font-size:100%; text-align:left; background-image: url(https://i.postimg.cc/sXwGWcwC/download.jpg); background-size: cover">

<h4 align="left"><span style="font-weight:700; font-size:150%"><font color=#d10202>Author:</font><font color=navy> Nima Pourmoradi</font></span></h4>
<h6 align="left"><font color=#ff6200><a href='https://github.com/NimaPourmoradi'>github: https://github.com/NimaPourmoradi</font></h6>
<h6 align="left"><font color=#ff6200><a href='https://www.kaggle.com/nimapourmoradi'>kaggle : https://www.kaggle.com/nimapourmoradi</a></font></h6>
<h6 align="left"><font color=#ff6200><a href='https://www.linkedin.com/in/nima-pourmoradi-081949288/'>linkedin : www.linkedin.com/in/nima-pourmoradi</a></font></h6>
<h6 align="left"><font color=#ff6200><a href='https://t.me/Nima_Pourmoradi'>Telegram : https://t.me/Nima_Pourmoradi</a></font></h6>

<img src="https://i.postimg.cc/t4b3WtCy/1000-F-291522205-Xkrm-S421-Fj-SGTMR.jpg">

<div style="background-color:#c5d8d1; padding: 25px 0px 10px 0px; border-radius: 10px; box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.1);border:0px solid #0A2342; text-align:center">
    <p style="font-size:18px; font-family:tahoma; line-height: 2em; text-indent: 20px;"><b>✅ If you like my notebook, please upvote it ✅
    </b></p>
</div>

##### [🏠 Tabel of Contents](#tbl_content)
