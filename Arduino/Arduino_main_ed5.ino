#include <Servo.h>
#include <Ultrasonic.h>

const int relay1 = 35; // relay of motor 5V
const int relay2 = 37; // relay of magnet 12V
const int Armmotor1 = 7; // the pin of arm motor1 (手臂左邊的馬達)
const int Armmotor2 = 8; // the pin of arm motor2 (手臂右邊的馬達)
Servo Armservo1; // 定義伺服馬達物件
Servo Armservo2;

String cmd;
int pwm_L=0,pwm_R=0,drift_time_L;

int ledA_R=34, ledA_G=46, ledA_B=36;  // left-front light, pwm pins
int ledB_R=38, ledB_G=40, ledB_B=42;  // right-front light, pwm pins
int ledC_R=26, ledC_G=28, ledC_B=30;  // left-rear light, digital pins
int ledD_R=27, ledD_G=29, ledD_B=31;  // right-rear light, digital pins
char pwmL[4],pwmR[4],sender,state,power,motion,command;
char ledA, ledB, ledC, ledD;
int Lf=10, Lb=11, Rf=12, Rb=13;  // Arduino mega pins


Ultrasonic ultrasonicF(45,47); // Front ultra - Trig 45, Echo 47
int distanceD,distanceF;

void setup() {
  Serial.begin(9600);
  runMotor(0,0);
  Armservo1.attach(Armmotor1); // need PWM pin
  Armservo2.attach(Armmotor2); // need PWM pin
  pinMode(relay1,OUTPUT);
  pinMode(relay2,OUTPUT);
  digitalWrite(relay1,LOW); // open the road of relay of motor
  digitalWrite(relay2,LOW); // open the road of relay of magnet
  Serial.println("Let's GO!"); // 這是用來確定 setup 已經跑完了
}

void loop() {
  
  if (Serial.available()) {
    //Serial.println("serial available");
    // 讀取傳入的字串直到'e'結尾 不包括e
    cmd = Serial.readStringUntil('e');
    //Serial.println(cmd); 

    sender = cmd[0];
    //Serial.println(sender);
    
    state = cmd[1];
    //Serial.println(state);
    
    power = cmd[2];
    //Serial.println(power);
 
    motion = cmd[3];
    //Serial.println(motion);

    command = motion;  // Arm commands
    
    if(sender=='1'){   // the signal is from python
      
      if(state=='1'){  // 甲關
          
        ledA = cmd[10];
        ledB = cmd[11];
        ledC = cmd[12];
        ledD = cmd[13];
        
        igniteLED(ledA_R,ledA_G,ledA_B,ledA);  // ledA ignited
        igniteLED(ledB_R,ledB_G,ledB_B,ledB);  // ledB ignited
        
        igniteLED(ledC_R,ledC_G,ledC_B,ledC);  // ledC ignited
        igniteLED(ledD_R,ledD_G,ledD_B,ledD);  // ledD ignited
        
        if(motion=='L'){  // Left : forward - Left drift - forward - Left drift - forward

          resetLED();
          // stop
          igniteLED(ledA_R,ledA_G,ledA_B,'2');
          igniteLED(ledB_R,ledB_G,ledB_B,'2');
          igniteLED(ledC_R,ledC_G,ledC_B,'2');
          igniteLED(ledD_R,ledD_G,ledD_B,'2');
          runMotor(0,0);
          delay(4000);

          resetLED();
          // forward 1, Red lights
          igniteLED(ledA_R,ledA_G,ledA_B,'7');
          igniteLED(ledB_R,ledB_G,ledB_B,'7');
          igniteLED(ledC_R,ledC_G,ledC_B,'7');
          igniteLED(ledD_R,ledD_G,ledD_B,'7');
          runMotor(35,35);
          delay(4000);

          drift_time_L = 6500;
          runMotor(5,120);  // to left, right wheel is faster
          resetLED();
          // left drift 1, left Yellow lights
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          
          delay(drift_time_L / 5);
          resetLED();
          delay(drift_time_L / 5);
          
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          
          delay(drift_time_L / 5);
          resetLED();
          delay(drift_time_L / 5);
          
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          
          delay(drift_time_L / 5);
          resetLED();

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 2, Cyan lights
          igniteLED(ledA_R,ledA_G,ledA_B,'5');
          igniteLED(ledB_R,ledB_G,ledB_B,'5');
          igniteLED(ledC_R,ledC_G,ledC_B,'5');
          igniteLED(ledD_R,ledD_G,ledD_B,'5');
          runMotor(60,60);
          delay(4000);

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // left drift 2, left Yellow lights
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledB_R,ledB_G,ledB_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          igniteLED(ledD_R,ledD_G,ledD_B,'6');
          runMotor(5,120);  // to left, right wheel is faster
          delay(6500);

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 3, Red lights
          igniteLED(ledA_R,ledA_G,ledA_B,'2');
          igniteLED(ledB_R,ledB_G,ledB_B,'2');
          igniteLED(ledC_R,ledC_G,ledC_B,'2');
          igniteLED(ledD_R,ledD_G,ledD_B,'2');
          runMotor(60,60);
          delay(3000);

          // stop
          runMotor(0,0);
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          delay(3000);
        }
        else if(motion=='R'){  // turn Right immediately

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 1, Red lights
          igniteLED(ledA_R,ledA_G,ledA_B,'2');
          igniteLED(ledB_R,ledB_G,ledB_B,'2');
          igniteLED(ledC_R,ledC_G,ledC_B,'2');
          igniteLED(ledD_R,ledD_G,ledD_B,'2');
          runMotor(0,0);
          delay(4000);
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 1, Red lights
          igniteLED(ledA_R,ledA_G,ledA_B,'7');
          igniteLED(ledB_R,ledB_G,ledB_B,'7');
          igniteLED(ledC_R,ledC_G,ledC_B,'7');
          igniteLED(ledD_R,ledD_G,ledD_B,'7');
          runMotor(35,35);
          delay(4000);

          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // right drift 1, right Yellow lights
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledB_R,ledB_G,ledB_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          igniteLED(ledD_R,ledD_G,ledD_B,'6');
          runMotor(120,5);
          delay(6500);

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 2, Cyan lights
          igniteLED(ledA_R,ledA_G,ledA_B,'5');
          igniteLED(ledB_R,ledB_G,ledB_B,'5');
          igniteLED(ledC_R,ledC_G,ledC_B,'5');
          igniteLED(ledD_R,ledD_G,ledD_B,'5');
          runMotor(60,60);
          delay(4000);

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // right drift 2, right Yellow lights
          igniteLED(ledA_R,ledA_G,ledA_B,'6');
          igniteLED(ledB_R,ledB_G,ledB_B,'6');
          igniteLED(ledC_R,ledC_G,ledC_B,'6');
          igniteLED(ledD_R,ledD_G,ledD_B,'6');
          runMotor(120,5);
          delay(6500);

          
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          // forward 3, Red lights
          igniteLED(ledA_R,ledA_G,ledA_B,'2');
          igniteLED(ledB_R,ledB_G,ledB_B,'2');
          igniteLED(ledC_R,ledC_G,ledC_B,'2');
          igniteLED(ledD_R,ledD_G,ledD_B,'2');
          runMotor(60,60);
          delay(3000);

          // stop
          runMotor(0,0);
          igniteLED(ledA_R,ledA_G,ledA_B,'0');
          igniteLED(ledB_R,ledB_G,ledB_B,'0');
          igniteLED(ledC_R,ledC_G,ledC_B,'0');
          igniteLED(ledD_R,ledD_G,ledD_B,'0');
          delay(3000);
        }
        else{
        
          if(power=='1'){  // power on, read pwm for motor
            pwmL[0] = cmd[4];
            pwmL[1] = cmd[5];
            pwmL[2] = cmd[6];
            pwmL[3] = '\0';
            pwm_L = atoi(pwmL);
            //Serial.println((String)pwm_L);
            Serial.println("pwm L:"+(String)pwm_L);
            
            pwmR[0] = cmd[7];
            pwmR[1] = cmd[8];
            pwmR[2] = cmd[9];
            pwmR[3] = '\0';
            pwm_R = atoi(pwmR);
            //Serial.println(pwm_R);
            Serial.println("pwm R:"+(String)pwm_R);
          }
          else{     // power off
            pwm_L = 0;
            pwm_R = 0;
          }

          runMotor(pwm_L,pwm_R);
          delay(10);
          
        }
        
     } 
     if(state=='2'){  // 乙 AB關
      
        ledA = cmd[10];
        ledB = cmd[11];
        ledC = cmd[12];
        ledD = cmd[13];
        motion = cmd[3];
        Serial.println("Received!");
        Serial.println(motion);
        
         
        igniteLED(ledA_R,ledA_G,ledA_B,ledA);  // ledA ignited
        igniteLED(ledB_R,ledB_G,ledB_B,ledB);  // ledB ignited
        igniteLED(ledC_R,ledC_G,ledC_B,ledC);  // ledC ignited
        igniteLED(ledD_R,ledD_G,ledD_B,ledD);  // ledD ignited
       
        if(motion=='E'){   // 放水果(B)
          while(1){
            distanceF = ultrasonicF.read();
            Serial.println(distanceF);
            if(distanceF < 65 && motion == 'E'){
                runMotor(0,0);
                igniteLED(ledA_R,ledA_G,ledA_B,'3');  // ledA ignited
                igniteLED(ledB_R,ledB_G,ledB_B,'3');  // ledB ignited
                igniteLED(ledC_R,ledC_G,ledC_B,'3');  // ledC ignited
                igniteLED(ledD_R,ledD_G,ledD_B,'3');  // ledD ignited
                delay(10000); // to be decided (ms)
                motion = 'B';
            }
            if(distanceF < 65 && motion=='B'){   // 所有任務已完成剩下擋板前停止(B)
              runMotor(20,20);
              if(distanceF < 10){
               runMotor(0,0);
               igniteLED(ledA_R,ledA_G,ledA_B,'2');  // ledA ignited
               igniteLED(ledB_R,ledB_G,ledB_B,'2');  // ledB ignited
               igniteLED(ledC_R,ledC_G,ledC_B,'2');  // ledC ignited
               igniteLED(ledD_R,ledD_G,ledD_B,'2');  // ledD ignited
               delay(100000); // to be decided (ms)
          }
          }
        
        
          }
         
        }
        
          }
          
          
        }

      if(command=='g'){   // 有任務已完成剩下擋板前停止(A)
          Serial.println("Received!!!!! ``````````````````````````````");
          while(1){
            distanceF = ultrasonicF.read();
            Serial.println(distanceF);
            runMotor(30,30);
            if(distanceF < 10){
                Serial.println("-------- STOP --------");
                runMotor(0,0);
                igniteLED(ledA_R,ledA_G,ledA_B,'3');  // ledA ignited
                igniteLED(ledB_R,ledB_G,ledB_B,'3');  // ledB ignited
                igniteLED(ledC_R,ledC_G,ledC_B,'3');  // ledC ignited
                igniteLED(ledD_R,ledD_G,ledD_B,'3');  // ledD ignited
                delay(100000); // to be decided (ms)
          }
        if(command == 'c') // c / close the circuit of relay
      {
        digitalWrite(relay1,HIGH); // open the relay of motor
        Serial.println("Open the relay of motor");
      }
        if(command == 'm') // m / close the circuit of magnet
      {
        digitalWrite(relay2,HIGH); // open the relay of motor
        Serial.println("Open the relay of magnet");
      }
        if(command == 's') // s / stop the Armmotor
      { // 數值 90 代表停止馬達轉動
        Armservo1.write(90);
        Armservo2.write(90);
        Serial.println("Stop work");
      }
        if(command == 'l') // l / turn left 手臂往左轉，往上轉
      {// 90 以上到 180 為順時針轉動，數字越大轉速越快
        Armservo1.write(110);
        Armservo2.write(70);
        Serial.println("Go Left");
      }
        if(command == 'k') //k, 手臂往下時撐住他
      {
        Armservo1.write(80);
        Armservo2.write(100);
        Serial.println("Fix arm");
      }  
        if(command == 'r') // r / turn right 手臂往右轉，往下動
      {// 90 以下到 0 為逆時針轉動，數字越小轉速越大
        Armservo1.write(90);
        Armservo2.write(135);
        Serial.println("Go Right");
        // Armservo1.write(90);
        // Armservo2.write(90);
      }
        if(command == 'o') // o / open the circuit of motor of relay 
      {
        digitalWrite(relay1,LOW); // open the relay of motor
        Serial.println("stop the power of motor(close the power source)");
      }
        if(command == 'p') // p / open the circuit of magnet of relay 
      {
        digitalWrite(relay2,LOW); // open the relay of motor
        Serial.println("stop the power of magnet (close the power source)");
      }
        
        
        if(power=='1'){  // power on
            pwmL[0] = cmd[4];
            pwmL[1] = cmd[5];
            pwmL[2] = cmd[6];
            pwmL[3] = '\0';
            pwm_L = atoi(pwmL);
            //Serial.println("pwm L:"+(String)pwm_L);
            
            pwmR[0] = cmd[7];
            pwmR[1] = cmd[8];
            pwmR[2] = cmd[9];
            pwmR[3] = '\0';
            pwm_R = atoi(pwmR);
            //Serial.println("pwm R:"+(String)pwm_R);
          }
          else{     // power off
            pwm_L = 0;
            pwm_R = 0;
          }

          runMotor(pwm_L,pwm_R);
          delay(50);
          //Serial.flush();
     }
   }   
  }
  else{
      delay(100);
  }
}

void LED_output(){

  pinMode(ledA_R, OUTPUT);
  pinMode(ledA_G, OUTPUT);
  pinMode(ledA_B, OUTPUT);
  
  pinMode(ledB_R, OUTPUT);
  pinMode(ledB_G, OUTPUT);
  pinMode(ledB_B, OUTPUT);
  
  pinMode(ledC_R, OUTPUT);
  pinMode(ledC_G, OUTPUT);
  pinMode(ledC_B, OUTPUT);
  
  pinMode(ledD_R, OUTPUT);
  pinMode(ledD_G, OUTPUT);
  pinMode(ledD_B, OUTPUT);
 
}

void igniteLED(int R,int G,int B,char choice){
  switch(choice){
    case '0':
      light(R,G,B,0,0,0);  // light off
      break;
      
    case '1':
      light(R,G,B,255,255,255);  // white
      break;
 
    case '2':
      light(R,G,B,255,0,0);  // Red
      break;

    case '3':
      light(R,G,B,0,255,0);  // Green
      break;
     
    case '4':
      light(R,G,B,0,0,255);  // Blue
      break;
    
    case '5':
      light(R,G,B,0,255,255);  // Cyan
      break;
      
    case '6':
      light(R,G,B,255,255,0);  // Yellow
      break;
      
    case '7':
      light(R,G,B,255,0,255);  // Megenta
      break;
      
  }
}

int get_pins(char key){
    if(key=='A'){
        return ledA_R, ledA_G, ledA_B;
    }
    if(key=='B'){
        return ledB_R, ledB_G, ledB_B;
    }
    if(key=='C'){
        return ledC_R, ledC_G, ledC_B;
    }
    if(key=='D'){
        return ledD_R, ledD_G, ledD_B;
    }
    
}


void resetLED(){
     
     igniteLED(ledA_R,ledA_G,ledA_B,'0');
     igniteLED(ledB_R,ledB_G,ledB_B,'0');
     
     igniteLED(ledC_R,ledC_G,ledC_B,'0');
     igniteLED(ledD_R,ledD_G,ledD_B,'0');
     
     digitalWrite(ledC_R,LOW);
     digitalWrite(ledC_G,LOW);
     digitalWrite(ledC_B,LOW);

     digitalWrite(ledD_R,LOW);
     digitalWrite(ledD_G,LOW);
     digitalWrite(ledD_B,LOW);
}

void runMotor(int pwm_L, int pwm_R){
  analogWrite(Lf,pwm_L);
  analogWrite(Lb,0);
  analogWrite(Rf,pwm_R);
  analogWrite(Rb,0);
}


void light(int R,int G,int B,int r,int g,int b){
  pinMode(R, OUTPUT);
  pinMode(G, OUTPUT);
  pinMode(B, OUTPUT);
  if(R==ledA_R || R==ledB_R){
    digitalWrite(R,255-r);
    digitalWrite(G,255-g);
    digitalWrite(B,255-b);
    
  }
  else{
    if(r==0){
      digitalWrite(R,LOW);
    }
    else{
      digitalWrite(R,HIGH);
    }
    if(g==0){
      digitalWrite(G,LOW);
    }
    else{
      digitalWrite(G,HIGH);
    }
    if(b==0){
      digitalWrite(B,LOW);
    }
    else{
      digitalWrite(B,HIGH);
    }
    
  }
}



 
