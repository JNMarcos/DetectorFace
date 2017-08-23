#include <opencv2\opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

string caminhoImagem = "Eu.jpg";
Mat deteccaoPele;
Mat deteccaoCabelo;
Mat imgQuantizadaPele;
Mat imgQuantizadaCabelo;

//constantes
const double VALOR_NAO_BRANCO = 0.001;
const int N_GRAUS = 360;
const int LIMIAR_QUANTIZACAO = 12;
const int TAM_MALHA_QUANTIZACAO = 5;
const double TAM_AREA_FACE = 12300;

//para a pele
const int H_LIM_SUP_PELE = 20;
const int H_LIM_INF_PELE = 240;

//para o cabelo
const int I_LIM_SUPERIOR = 80;
const int LIM_BGR_CABELO = 15;
const int H_LIM_SUP_CABELO = 40;
const int H_LIM_INF_CABELO = 20;


//calcula o limite superior
//dado r (R normalizado)
double f1(double r) {
	return -1.376 * r * r + 1.0743 * r + 0.2;
}

//calcula o limite inferior
//dado r (R normalizado)
double f2(double r) {
	return -0.77 * r * r + 0.5601 * r + 0.18;
}

//calcula o valor de w = white
//dado r e g (R e G normalizados respectivamente)
double w(double r, double g) {
	return pow(r - 0.33, 2) + pow(g - 0.33, 2);
}

//calcula o valor de arcocosseno
//dado R, G, B
double arcocosseno(int R, int G, int B) {
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
	//o H já foi calculado em isPele
	bool isCabelo = false;
	if ((I < I_LIM_SUPERIOR) &&
			((B - G < LIM_BGR_CABELO) || (B - R < LIM_BGR_CABELO))
				|| (H > H_LIM_INF_CABELO && H <= H_LIM_SUP_CABELO)) {
		isCabelo = true;
	}
	return isCabelo;
}

Mat detectarComponentes(Mat imagem) {
	//cvtColor(imagem, imagem, CV_BGR2GRAY);
	//suaviza a imagem, removendo pontos. Tamanho da malha de suavização: 3x3
	blur(imagem, imagem, Size(3, 3));
	//binariza a imagem, invertendo
	threshold(imagem, imagem, 120, 255, CV_THRESH_BINARY_INV);

	vector<Point> vertices;
	vector< vector <Point>> contornos;
	vector<Vec4i> hierarquia;

	double areaMaxima = 0;
	double area;
	int index = 0;
	//encontra os contornos
	//findContours(imagem.clone(), contornos, hierarquia, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	findContours(imagem.clone(), contornos, hierarquia, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	Mat desenhoContorno = Mat::zeros(imagem.size(), CV_8UC1);
	
	for (int i = 0; i < contornos.size(); i++) {
		area = contourArea(contornos[i]);
		cout << area;
		if (area > areaMaxima) {
			areaMaxima = area;
			index = i;
		}

		Scalar cor(rand() % 255);
		drawContours(desenhoContorno, contornos, i, cor);
	}

	

	cout << "\n" << areaMaxima;

	/*if (contornos.size() != 0){
		for (int i = 0; i >= 0; i = hierarquia[i][0]) {
			area = contourArea(contornos[i]);
			cout << area;
			if (area > areaMaxima) {
				areaMaxima = area;
			}
			Scalar cor(rand() % 255);
			drawContours(desenhoContorno, contornos, i, cor);
		}
	}*/
	return desenhoContorno;
}

//um mesmo método para quantização
//se isPele = true é quantização da pele, caso isPele = 0 é quantização do cabelo
void quantizar(Mat imagemPele, Mat imagemCabelo) {
	//nº de linhas e colunas da imagem quantizada
	int nLinhasQ = (int) (imagemPele.rows / TAM_MALHA_QUANTIZACAO);
	int nColunasQ = (int) (imagemPele.cols / TAM_MALHA_QUANTIZACAO);
	int nPixelsMalha = pow(TAM_MALHA_QUANTIZACAO, 2);

	imgQuantizadaPele = Mat::ones(nLinhasQ, nColunasQ, CV_8UC1);
	imgQuantizadaCabelo = Mat::ones(nLinhasQ, nColunasQ, CV_8UC1);
	
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

	int somaN;

	int nColunas = imagem.cols;
	int nLinhas = imagem.rows;

	//matrizes de apenas um canal
	deteccaoPele = Mat::zeros(nLinhas, nColunas, CV_8UC1);
	deteccaoCabelo = Mat::zeros(nLinhas, nColunas, CV_8UC1);
	
	if (imagem.isContinuous()) {
		nColunas *= nLinhas;
		nLinhas = 1;
	}
	
	//para ter uma divisão em canais (opção 2)
	//vector<Mat> BGR;
	for (int i = 0; i < nLinhas; i++) {
		const uchar* Mi = imagem.ptr<uchar>(i);
		for (int j = 0; j < nColunas; j++) {
			//falta botar i e j aqui
			bgr = imagem.at<Vec3b>(100, 9);

			/*
			//divide a imagem em planos, no caso, 3
			split(imagem, BGR);
			azul = (int)BGR[0].at<uchar>(1, 1);
			verde = (int)BGR[1].at<uchar>(1, 1);
			vermelho = (int)BGR[2].at<uchar>(1, 1);
			*/

			//pega a intensidade dos valores BGR
			B = (int) bgr.val[0];
			G = (int)bgr.val[1];
			R = (int) bgr.val[2];

			somaN = B + G + R;

			r = R / somaN;
			g = G / somaN;

			H = h(R, G, B);
			I = intensidade(R, G, B);

			cout << B << " verde " << G << " verm " << R << " soma = " << B + G + R;	

			//se é true, então é pele, logo põe 1
			if (isPele(R, G, B, r, g, H)) {
				deteccaoPele.at<uchar>(i, j) = 1;
			}

			//se true, é cabelo, logo põe 1
			if (isCabelo(R, G, B, H, I)) {
				deteccaoCabelo.at<uchar>(i, j) = 1;
			}
		}
	}
}

int main() {
	//imagem a ser usada para detectar rostos
	//CV_8UC3
	Mat imgEntrada = imread(caminhoImagem, 1);
	/*teste: até split*/
	vector<Mat> BGR;
	split(imgEntrada, BGR);
	
	//pipeline
	detectar(imgEntrada);
	quantizar(deteccaoPele, deteccaoCabelo);

	Mat desenhoComp = detectarComponentes(BGR[2]);

	//encontra-se áreas mínimas 
	RotatedRect encaixotamentoPele = minAreaRect(imgQuantizadaPele);
	RotatedRect encaixotamentoCabelo = minAreaRect(imgQuantizadaCabelo);

	vector<Point> ptsIntersecaoPeleCabelo;
	//verifica se as áreas estão intersectadas
	int resultadoIntersecao = rotatedRectangleIntersection(encaixotamentoPele, encaixotamentoCabelo, ptsIntersecaoPeleCabelo);
	
	//se ao menos há uma interseção 
	if (resultadoIntersecao != 0) {
		if (ptsIntersecaoPeleCabelo.size() == 2 ||
			ptsIntersecaoPeleCabelo.size() == 4 ||
			ptsIntersecaoPeleCabelo.size() == 8) {
				
		}
	}


	//cout << "\n\n\nimagemNormal\n" << imgNormalizada;
	imwrite("approxComponenteCanalVermelho.png", desenhoComp);
	imshow("imgEntrada", desenhoComp);
	waitKey();

	return 0;
}