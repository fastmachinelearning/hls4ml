pipeline {
  agent {
    docker {
      image 'hls4ml-with-vivado:latest'
    }
  }
  options {
    timeout(time: 3, unit: 'HOURS')
  }

  stages {
    stage('Keras to HLS') {
      parallel {
        stage('py36') {
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 . activate hls4ml-py36
                 cat keras-models.txt | xargs ./keras-to-hls.sh -p 3
                 cat keras-models-serial.txt | xargs ./keras-to-hls.sh -p 3 -s'''
            }
          }
        }
        stage('py27') {
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 . activate hls4ml-py27
                 cat keras-models.txt | xargs ./keras-to-hls.sh -p 2
                 cat keras-models-serial.txt | xargs ./keras-to-hls.sh -p 2 -s'''
            }
          }
        }
      }
      post {
        success {
          dir(path: 'test') {
              sh '''#!/bin/bash
                 ./py-diff.sh -r 2'''
          }
        }
      }
    }
    stage('C Simulation') {
      parallel {
        stage('2017.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2017', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2017.2 -c -p 2'''
            }
          }
        }
        stage('2018.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2018', value: '1';
              environment name: 'TEST_SIMULATION', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2018.2 -c -p 2'''
            }
          }
        }
      }
    }
    stage('C/RTL Synthesis') {
      parallel {
        stage('2017.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2017', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2017.2 -r -p 2'''
            }
          }
        }
        stage('2018.2') {
          when {
            allOf {
              environment name: 'USE_VIVADO_2018', value: '1';
              environment name: 'TEST_SYNTHESIS', value: '1'
            }
          }
          steps {
            dir(path: 'test') {
              sh '''#!/bin/bash
                 ./build-prj.sh -i /opt/Xilinx -v 2018.2 -r -p 2'''
            }
          }
        }
      }
    }
    stage('Report') {
      when {
        environment name: 'TEST_SYNTHESIS', value: '1'
      }
      steps {
        dir(path: 'test') {
          sh '''#!/bin/bash
             ./gather-reports.sh -b'''
        }

      }
    }
  }
  post {
    always {
      dir(path: 'test') {
          sh '''#!/bin/bash
             ./cleanup.sh'''
      }
    }
  }
}
