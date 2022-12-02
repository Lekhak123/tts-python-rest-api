import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN
from flask import Flask, request, jsonify, send_file
app = Flask(__name__)
tacotron2 = Tacotron2.from_hparams(source="./tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="./tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

@app.route('/tts', methods=['POST'])
def tts():
    json = request.get_json()
    text = json.get('text') 
    if not text:
        return "text field is required"
    else:
        try:
            mel_output, mel_length, alignment = tacotron2.encode_text(text)
            waveforms = hifi_gan.decode_batch(mel_output)
            path_to_file= "./audio/"+text+".wav"
            path_to_file = path_to_file.replace(" ","")
            torchaudio.save(path_to_file,waveforms.squeeze(1), 22050)
            return send_file(
                path_to_file, 
                mimetype="audio/wav", 
                as_attachment=True, 
                download_name=path_to_file)
        except Exception as error:
            print(error)




if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True)
    








