const lstm = ml5.charRNN('models/edgar_allan_poe_edited', modelReady);

let inputText = "";
let inputTextDom;
let modelStateDom;
let outputTextDom;
let start_pause_button;
let reset_button;
let generating = false;

function setup(){
    noCanvas();

    modelStateDom = select('#modelState');
    modelStateDom.html('model loading...🥚');
    inputTextDom = select('#inputText');
    inputText = inputTextDom.html();

    start_pause_button = createButton('run🏃‍♀️');
    start_pause_button.mousePressed(generate);

    reset_button = createButton('reset🤸‍♂️');
    reset_button.mousePressed(resetModel);

    outputTextDom = createDiv('');
    outputTextDom.id('outputText');
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
  outputTextDom.html(next.sample,true);
}