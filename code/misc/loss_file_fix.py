
for i in range(5):
    loss_file = 'run_' + str(i)+'/model_val_loss.txt'
    file1 = open(loss_file, 'r')
    lines = file1.readlines()
    loss_list = []
    count = 0

    for line in lines:
        if 'model_info' in line:
            model_info = line.replace('model_info: ','').replace('\n', '')
            new_file = 'run_' + str(i) + '/'+model_info +'_model_val_loss.txt'

            f = open(new_file, 'a')
        if 'val_loss' in line:
            loss = line.split('val_loss: ')[1]
            loss = loss.split('\n')[0]
            print('loss: ', loss)
            loss = float(loss)
            f.write('val_loss: '+str(loss))
            f.write('\n\n')
            f.close()

        if count==4:
            count=0