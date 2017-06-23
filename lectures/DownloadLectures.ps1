$client = new-object System.Net.WebClient
$client.Credentials =  Get-Credential

$sourceFolder = 'http://cs188websitecontent.s3.amazonaws.com/lectures/'
$destFolder = 'C:\Users\jyesa\Desktop\home\jye\docs\ai_edex\lectures\'

for ($i=1;$i -le 26;$i++){ 
    $client.DownloadFile($sourceFolder+'fa13-cs188-lecture-'+$i+'-1PP.pdf', 
    $destFolder+'lecture-'+$i+'.pdf')
    }