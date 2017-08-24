#include <opencv2\opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

string caminhoImagem = "Modelo1.jpg";
Mat imgDeteccaoPele;
Mat imgDeteccaoCabelo;
Mat imgQuantizadaPele;
Mat imgQuantizadaCabelo;
Mat imgComponentePele;
Mat imgComponenteCabelo;

RotatedRect encaixotamentoPele;
RotatedRect encaixotamentoCabelo;

//constantes
const double VALOR_NAO_BRANCO = 0.001;
const double N_GRAUS = 2 * 3.141592653589793238463;
const int LIMIAR_QUANTIZACAO = 12;
const int TAM_MALHA_QUANTIZACAO = 5;
const double TAM_AREA_FACE = 30;
const int VAL_MAX_CORES = 255;
const int TAM_BLUR = 3;

//para a pele
const double H_LIM_SUP_PELE = 0.60;// antes era 0.349066;
const double H_LIM_INF_PELE = 4.18879;

//para o cabelo
const int I_LIM_SUPERIOR = 50; //antes era 80
const int I_LIM_BRANCO = 20; //não existia
const int LIM_BGR_CABELO = 65;
const double H_LIM_SUP_CABELO = 0.698132;
const double H_LIM_INF_CABELO = 0.349066;


//calcula o limite superior
//dado r (R normalizado)
double f1(double r) {
	return -1.376 * pow(r, 2) + 1.0743 * r + 0.2;
}

//calcula o limite inferior
//dado r (R normalizado)
double f2(double r) {
	return -0.776 * pow(r, 2) + 0.5601 * r + 0.18;
}

//calcula o valor de w = white
//dado r e g (R e G normalizados respectivamente)
double w(double r, double g) {
	return pow(r - 0.33, 2) + pow(g - 0.33, 2);
}

//calcula o valor de arcocosseno
//dado R, G, B
double arcocosseno(int R, int G, int B) {
	if (R == G && G == B)
		return 200; //mantém condição segura
	else
		return acos((0.5 * ((R - G) + (R - B))) / sqrt(pow(R - G, 2) + (R - B) * (G - B)));
}

//calcula o valor de H
//dado R, G, B (usa-se a função arcocosseno)
double h(int R, int G, int B) {
	double ang = arcocosseno(R, G, B);
	double H;
	if (B <= G) H = ang;
	else H = N_GRAUS - ang;
	return H;
}

//calcula a intensidade
//dado R, G e B
double intensidade(int R, int G, int B) {
	return (R + G + B) / 3;
}

bool isPele(int R, int G, int B, double r, double g, double H) {
	bool isPele = false;
	if ((g < f1(r)) &&
		(g > f2(r)) &&
		(w(r, g) > VALOR_NAO_BRANCO) &&
		(H > H_LIM_INF_PELE || H <= H_LIM_SUP_PELE)) {
		isPele = true;
	}

	return isPele;
}

bool isCabelo(int R, int G, int B, double H, double I) {
	bool isCabelo = false;
	if ((I < I_LIM_SUPERIOR) &&
			((B - G < LIM_BGR_CABELO) || (B - R < LIM_BGR_CABELO))
		|| (H > H_LIM_INF_CABELO && H <= H_LIM_SUP_CABELO
			&& I < I_LIM_BRANCO)
				) {
		isCabelo = true;
	}
	return isCabelo;
}

Mat detectarComponentes(Mat imagem, bool isPele) {
	//cvtColor(imagem, imagem, CV_BGR2GRAY);
	//suaviza a imagem, removendo pontos. Tamanho da malha de suavização: TAM_BLURXTAM_BLUR
	//blur(imagem, imagem, Size(TAM_BLUR, TAM_BLUR));
	//binariza a imagem, invertendo
	//threshold(imagem, imagem, 120, VAL_MAX_CORES, CV_THRESH_BINARY);

	vector<Point> vertices;
	vector< vector <Point>> contornos;
	Mat oContorno;
	vector<Vec4i> hierarquia;

	double areaMaxima = 0;
	double area;
	int index = 0;
	Mat desenhoContorno = Mat::zeros(imagem.size(), CV_8UC1);


	//encontra os contornos
	findContours(imagem.clone(), contornos, hierarquia, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (contornos.size() != 0){
		for (int i = 0; i >= 0; i = hierarquia[i][0]) {
			area = contourArea(contornos[i]);
			cout << area << "\n";
			//if (area > TAM_AREA_FACE){
				if (area > areaMaxima) {
					areaMaxima = area;
					index = i;
				}
				
				/*Scalar cor(rand() % 255);
				drawContours(desenhoContorno, contornos, i, cor);
				oContorno.push_back(contornos[i]);*/
			//}
		}
	}

	Scalar cor(rand() % 255);
	drawContours(desenhoContorno, contornos, index, cor);
	oContorno.push_back(contornos[index]);

	if (isPele) {
		imgComponentePele = desenhoContorno;
		encaixotamentoPele = minAreaRect(contornos[index]);
	}
	else {
		imgComponenteCabelo = desenhoContorno;
		encaixotamentoCabelo = minAreaRect(contornos[index]);
	}

	
	return oContorno;
}

//um mesmo método para quantização
//se isPele = true é quantização da pele, caso isPele = 0 é quantização do cabelo
void quantizar(Mat imagemPele, Mat imagemCabelo) {
	//nº de linhas e colunas da imagem quantizada
	int nLinhasQ = (int) (imagemPele.rows / TAM_MALHA_QUANTIZACAO);
	int nColunasQ = (int) (imagemPele.cols / TAM_MALHA_QUANTIZACAO);
	int nPixelsMalha = pow(TAM_MALHA_QUANTIZACAO, 2);

	imgQuantizadaPele = Mat::ones(nLinhasQ, nColunasQ, CV_8UC1) * VAL_MAX_CORES;
	imgQuantizadaCabelo = Mat::ones(nLinhasQ, nColunasQ, CV_8UC1) * VAL_MAX_CORES;
	
	Mat subMatrizPele, subMatrizCabelo;
	int nDePixelsNaoPele, nDePixelsNaoCabelo;
	//malha não sobreposta
	//se não houver numa iteração TAM_MALHA_QUANTIZACAO, essas linhas são descartadas
	//o mesmo princípio vale para 
	for (int i = 0; i <= imagemPele.rows - TAM_MALHA_QUANTIZACAO; i = i + TAM_MALHA_QUANTIZACAO) {
		for (int j = 0; j <= imagemPele.cols - TAM_MALHA_QUANTIZACAO; j = j + TAM_MALHA_QUANTIZACAO) {
			//pega-se submatriz das imagens
			subMatrizPele = imagemPele(Range(i, i + TAM_MALHA_QUANTIZACAO), Range(j, j + TAM_MALHA_QUANTIZACAO));
			subMatrizCabelo = imagemCabelo(Range(i, i + TAM_MALHA_QUANTIZACAO), Range(j, j + TAM_MALHA_QUANTIZACAO));
			
			//countNonZero conta o número de não zeros
			//como é binária, nos entrega exatamente o número de 1s
			nDePixelsNaoPele = nPixelsMalha - countNonZero(subMatrizPele);
			nDePixelsNaoCabelo = nPixelsMalha - countNonZero(subMatrizCabelo);

				if (nDePixelsNaoPele > LIMIAR_QUANTIZACAO) {
					imgQuantizadaPele.at<uchar>(i / TAM_MALHA_QUANTIZACAO, j / TAM_MALHA_QUANTIZACAO) = 0;
				}

				if (nDePixelsNaoCabelo > LIMIAR_QUANTIZACAO) {
					imgQuantizadaCabelo.at<uchar>(i / TAM_MALHA_QUANTIZACAO, j / TAM_MALHA_QUANTIZACAO) = 0;
				}
		}
	}
}


//detecta cabelo e pele
void detectar(Mat imagem) {
	//vetor de 3 canais
	Vec3b bgr;

	//valores de intensidade por canal
	int B;
	int G;
	int R;

	//valores normalizados
	double r;
	double g;

	//intensidade
	double H;
	double I;

	int somaBGR;

	int nColunas = imagem.cols;
	int nLinhas = imagem.rows;

	//matrizes de apenas um canal
	imgDeteccaoPele = Mat::zeros(nLinhas, nColunas, CV_8UC1);
	imgDeteccaoCabelo = Mat::zeros(nLinhas, nColunas, CV_8UC1);
	
	for (int i = 0; i < nLinhas; i++) {
		for (int j = 0; j < nColunas; j++) {
			bgr = imagem.at<Vec3b>(i,j);

			//pega a intensidade dos valores BGR
			B = (int) bgr.val[0];
			G = (int)bgr.val[1];
			R = (int) bgr.val[2];

			somaBGR = B + G + R;

			r = ((double) R / somaBGR);
			g = ((double) G / somaBGR);

			H = h(R, G, B);
			I = intensidade(R, G, B);

			//se é true, então é pele, logo põe VAL_MAX_CORES
			if (isPele(R, G, B, r, g, H)) {
				imgDeteccaoPele.at<uchar>(i, j) = VAL_MAX_CORES;
			}

			//se true, é cabelo, logo põe VAL_MAX_CORES
			if (isCabelo(R, G, B, H, I)) {
				imgDeteccaoCabelo.at<uchar>(i, j) = VAL_MAX_CORES;
			}
		}
	}
}

int main() {
	//imagem a ser usada para detectar rostos
	//CV_8UC3
	Mat imgEntrada = imread(caminhoImagem, 1);
	
	//pipeline
	detectar(imgEntrada);
	imshow("imgEntrada", imgEntrada);
	imwrite("imgPeleDetectada" + caminhoImagem, imgDeteccaoPele);
	imwrite("imgCabeloDetectado" + caminhoImagem, imgDeteccaoCabelo);

	quantizar(imgDeteccaoPele, imgDeteccaoCabelo);
	imwrite("peleQuantizada" + caminhoImagem, imgQuantizadaPele);
	imwrite("cabeloQuantizada" + caminhoImagem, imgQuantizadaCabelo);

	detectarComponentes(imgQuantizadaPele.clone(), true);
	imwrite("peleComponente" + caminhoImagem, imgComponentePele);

	detectarComponentes(imgQuantizadaCabelo.clone(), false);
	imwrite("cabeloComponente" + caminhoImagem, imgComponenteCabelo);

	vector<Point> ptsIntersecaoPeleCabelo;
	//verifica se as áreas estão intersectadas
	int resultadoIntersecao = rotatedRectangleIntersection(encaixotamentoPele, encaixotamentoCabelo, ptsIntersecaoPeleCabelo);
	
	//se ao menos há uma interseção 
	if (resultadoIntersecao != 0) {
		cout << "Pontos "<< ptsIntersecaoPeleCabelo.size();
		if (ptsIntersecaoPeleCabelo.size() == 2 ||
			ptsIntersecaoPeleCabelo.size() == 4 ||
			ptsIntersecaoPeleCabelo.size() == 6 ||
			ptsIntersecaoPeleCabelo.size() == 8) {
			cout << "\nDeu certo";
			Scalar cor(rand() % 255);
			vector< vector < Point > > pts;
			pts.push_back(ptsIntersecaoPeleCabelo);
			drawContours(imgDeteccaoPele, pts, 0, cor);
			imshow("Final", imgDeteccaoPele);
			waitKey();
		}
		else {
			cout << "Quase! Ajeita esse cabelo!";
		}
	}
	else {
		cout << "Nem ao menos há interseção.";
	}

	waitKey();
	return 0;
}