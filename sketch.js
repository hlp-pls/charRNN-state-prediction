const lstm = ml5.charRNN('models/edgar_allan_poe_edited', modelReady);

let inputText = "";
let canvas;
let inputTextDom;
let modelStateDom;
let outputTextDom;
let start_pause_button;
let reset_button;
let generating = false;

let probs = [];

let w = 30;
let h = 40;
   
let num_w;
let num_h;

function setup(){
    modelStateDom = select('#modelState');
    modelStateDom.html('model loading...🥚');
    inputTextDom = select('#inputText');
    inputText = inputTextDom.html();

    start_pause_button = createButton('run🏃‍♀️');
    start_pause_button.id('run_button');
    start_pause_button.mousePressed(generate);

    reset_button = createButton('reset🤸‍♂️');
    reset_button.id('reset_button');
    reset_button.mousePressed(resetModel);

    num_w = floor(windowWidth/w); 
    let cw = windowWidth;
    let ch = (floor(113/num_w)+1)*h;
    canvas = createCanvas(cw,ch);

    num_h = floor(height/h);

    outputTextDom = createSpan('');
    outputTextDom.id('outputText');
}

function draw(){

    background(255,255,0);
    
    for(let i=0; i<num_w; i++){
        for(let j=0; j<num_h; j++){
            
            let index = i + num_w * j;
            if(0<probs.length&&probs.length>index){
                fill(255,255,255*probs[index].probability);
                rect(w*i,h*j,w,h);
                fill(0);
                text(probs[index].char,w*(i)+5,h*(j)+15);
                text(floor(probs[index].probability*100),w*(i)+5,h*(j)+30);
            }else{
            
            }
        }
    }

}

function modelReady(){
    console.log("LSTM model loaded.");
    modelStateDom.html('model loaded🐣');
    resetModel();
}

function resetModel(){
    lstm.reset();
    const seed = inputTextDom.html();
    lstm.feed(seed);
    outputTextDom.html(seed);
}

function generate(){
    if (generating) {
        generating = false;
        start_pause_button.html('run🏃‍♀️');
    } else {
        generating = true;
        start_pause_button.html('pause🚶‍♂️');
        loopRNN();
    }
}

async function loopRNN() {
  while (generating) {
    await predict();
  }
}

async function predict() {
  let temperature = 0.5;
  let next = await lstm.predict( temperature );
  await lstm.feed(next.sample);
  //console.log(next.probabilities);
  probs = next.probabilities;
  outputTextDom.html(next.sample,true);
}