import urllib.parse
import urllib.request
import re
import pandas as pd
import os
url = 'https://www.uniprot.org/uploadlists/'

project_dir = '/home/tasnina/Projects/SynVerse/'
dirname = project_dir + '/datasets/networks/TissueNet/HPA-Protein/'

for infile in os.listdir(dirname):
    if not 'uniprot' in infile:
        infilepath = dirname + infile
        outfile = 'uniprot_' + infile
        outfilepath  = dirname + outfile
        print(infile)

        if not os.path.exists(outfilepath):
            ppi_df = pd.read_csv(infilepath, sep='\t', header=None, names=['p1', 'p2'], index_col=False)
            ensmbl_ids_1 = set(ppi_df['p1'])
            ensmbl_ids_2= set(ppi_df['p2'])
            ensmbl_ids = list(ensmbl_ids_1.union(ensmbl_ids_2))

            print(len(ensmbl_ids))

            start = 0

            response_list=[]

            from_id = []
            to_id = []
            while start<len(ensmbl_ids):
                print('start: ', start)
            # while start < 50:

                if start+500 < len(ensmbl_ids):
                    end = start+500
                else:
                    end = len(ensmbl_ids)

                params = {
                'from': 'ENSEMBL_ID',
                'to': 'ACC',
                'format': 'tab',
                'query': '\t'.join(ensmbl_ids[start:end])
                }

                data = urllib.parse.urlencode(params)
                data = data.encode('utf-8')
                req = urllib.request.Request(url, data)

                with urllib.request.urlopen(req) as f:
                   response = f.read()

                response = response.decode('utf-8')
                # print(response)
                # print(type(response))
                # response = response.split('\t')[2:]
                response = re.split(r"\t|\n", response)[2:-1]
                start = end


                for i in range(len(response)):
                    if i%2==0:
                        from_id.append(response[i])
                    else:
                        to_id.append(response[i])


            ensembl_to_uniport_df = pd.DataFrame({'ensembl':from_id, 'uniprot': to_id })

            #one ensembl id can be mapped to multiple uniprot. keep only the first one. checked for a few entry
            #that the first one is the reviewed one if there is a reviewed uniprot present for a ensembl_id.

            ensembl_to_uniport_df.drop_duplicates(subset=['ensembl'],inplace=True, keep='first')
            ensembl_to_uniport_dict = dict(zip(ensembl_to_uniport_df['ensembl'],\
                                                ensembl_to_uniport_df['uniprot']))
            # print(ensembl_to_uniport_dict)

            ppi_df = ppi_df[(ppi_df['p1'].isin(list(ensembl_to_uniport_df['ensembl']))) &
                            (ppi_df['p2'].isin(list(ensembl_to_uniport_df['ensembl'])))]

            # print(ppi_df.columns)
            # print(ppi_df['p1'])
            ppi_df['p1']= ppi_df['p1'].astype(str).apply(lambda x:ensembl_to_uniport_dict[x])
            ppi_df['p2']= ppi_df['p2'].astype(str).apply(lambda x:ensembl_to_uniport_dict[x])

            ppi_df = ppi_df[['p1','p2']]

            ppi_df.to_csv(outfilepath,sep='\t', index=False)

            print('done with: ', infile)

